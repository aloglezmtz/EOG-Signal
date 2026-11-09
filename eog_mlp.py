#!/usr/bin/env python3
"""Proyecto Final - Agrupamiento/clasificación de señales EOG (Blink-to-Speak) con MLP.

Autores: Alondra Goretti González Martínez (218534681) y Emiliano Barraza Cervantes (219425754)

Uso:  python eog_mlp.py            # pipeline completo (Fases 1-5)
      python eog_mlp.py --rapido   # menos repeticiones/rejilla (prueba rápida)
      python eog_mlp.py --csv otro.csv

Entradas : data/dataset_eog.csv (200 registros, 30 características + Clase)
Salidas  : resultados/*.csv, data/dataset_reducido.csv, figs/*.png, docs/data/resultados.json
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix, log_loss,
                             normalized_mutual_info_score, precision_recall_fscore_support, silhouette_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ROOT = Path(__file__).parent
SEED, THR = 42, 0.8
CLASES = ["1 Sí", "2 No", "3 Quiero dormir", "4 Agua", "5 Baño", "6 Quiero comer",
          "7 No estoy bien", "8 Falta de aire", "9 Dolor", "10 Peligro"]
# Recall por clase (%) reportado en el artículo (árbol de decisión, Fig. 9); promedio 83.4 %
ART_RECALL = [53.0, 88.5, 100.0, 95.5, 65.0, 79.5, 81.0, 88.5, 82.5, 100.0]
# Fig. 6 del artículo: se conservan 19 características (|r| < 0.8); las 11 restantes del dataset se eliminaron
ART_ELIMINADAS = ["SC_Std_Derecha", "SC_Std_Izquierda", "SC_Std_Horizontales", "SC_Multiplicacion_Std",
                  "SC_Suma_Std", "SC_Multiplicacion_RMS", "SC_Multiplicacion_Varianza", "SC_Suma_Varianza",
                  "SC_DifMaxMin_Derecha", "SC_DifMaxMin_Izquierda", "SC_DifMaxMin_Horizontales"]
C = dict(azul="#1F3A5F", verde="#2A9D8F", naranja="#E76F51", gris="#8D99AE", ambar="#E9C46A")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 130})


def corto(c):
    for a, b in [("SC_", ""), ("Derecha", "Der"), ("Izquierda", "Izq"), ("Horizontales", "Hor"),
                 ("Multiplicacion", "Mult"), ("DistanciaPicos", "DistPicos")]:
        c = c.replace(a, b)
    return c


def mlp_pipe(p, max_iter):
    return Pipeline([("sc", StandardScaler()),
                     ("mlp", MLPClassifier(solver="adam", max_iter=max_iter, random_state=SEED, **p))])


def evaluar(modelo, X, y, reps):
    """CV estratificada de 5 pliegues repetida `reps` veces (el artículo usó 10 repeticiones)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=SEED)

    def una(tr, te):
        m = clone(modelo).fit(X.iloc[tr], y.iloc[tr]); pr = m.predict(X.iloc[te])
        p, r, f, _ = precision_recall_fscore_support(y.iloc[te], pr, average="macro", zero_division=0)
        return accuracy_score(y.iloc[te], pr), p, r, f, confusion_matrix(y.iloc[te], pr, labels=range(1, 11))
    out = Parallel(n_jobs=-1)(delayed(una)(tr, te) for tr, te in cv.split(X, y))
    acc = np.array([o[0] for o in out]); cm = sum(o[4] for o in out)
    return dict(acc=acc, acc_rep=acc.reshape(reps, 5).mean(1), prec=np.mean([o[1] for o in out]),
                rec=np.mean([o[2] for o in out]), f1=np.mean([o[3] for o in out]),
                cm=(cm / cm.sum(1, keepdims=True) * 100))


def curvas(X, y, p, epocas):
    """Curva de aprendizaje por época (partial_fit) en 5 pliegues: pérdida y exactitud de train/validación."""
    cl = np.arange(1, 11); res = {k: [] for k in ("l_tr", "l_va", "a_tr", "a_va")}
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        sc = StandardScaler().fit(X.iloc[tr]); Xt, Xv = sc.transform(X.iloc[tr]), sc.transform(X.iloc[va])
        yt, yv = y.iloc[tr].values, y.iloc[va].values
        m = MLPClassifier(solver="adam", random_state=SEED, **p); c = {k: [] for k in res}
        for _ in range(epocas):
            m.partial_fit(Xt, yt, classes=cl)
            c["l_tr"].append(log_loss(yt, m.predict_proba(Xt), labels=cl))
            c["l_va"].append(log_loss(yv, m.predict_proba(Xv), labels=cl))
            c["a_tr"].append(m.score(Xt, yt)); c["a_va"].append(m.score(Xv, yv))
        for k in res: res[k].append(c[k])
    return {k: np.array(v) for k, v in res.items()}


def elegir_iter(cur):
    s = pd.Series(cur["a_va"].mean(0)).rolling(15, min_periods=1).mean()
    ep = int(np.argmax(s.values >= s.max() - 0.005)) + 1
    return int(max(100, np.ceil(ep / 50) * 50)), ep


def reducir_corr(X, thr=THR):
    """Del par con mayor |r| > thr se elimina la característica más redundante (mayor |r| medio)."""
    c = X.corr().abs(); keep = list(X.columns); quitadas = []
    while True:
        s = c.loc[keep, keep].values.copy(); np.fill_diagonal(s, 0)
        if s.max() <= thr: break
        i, j = np.unravel_index(s.argmax(), s.shape)
        d, o = (i, j) if s[i].mean() >= s[j].mean() else (j, i)
        quitadas.append((keep[d], keep[o], float(s[i, j]))); keep.pop(d)
    return keep, quitadas


def heat(ax, M, xl, yl, vmax=None, tam=7):
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax); ax.grid(False)
    ax.set_xticks(range(len(xl)), xl, rotation=60, ha="right", fontsize=tam); ax.set_yticks(range(len(yl)), yl, fontsize=tam)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 0.5:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=tam,
                        color="white" if M[i, j] > (vmax or M.max()) * .55 else "black")
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--csv", default=str(ROOT / "data/dataset_eog.csv"))
    a = ap.parse_args(); t0 = time.time(); reps = 3 if a.rapido else 10
    for d in ("figs", "resultados", "docs/data"): (ROOT / d).mkdir(parents=True, exist_ok=True)
    R = {}

    # ---------------- Fase 1: preprocesamiento ----------------
    df = pd.read_csv(a.csv); y = df["Clase"]; Xall = df.drop(columns="Clase"); X = Xall.drop(columns="Longitud")
    print(f"Dataset: {df.shape}, faltantes={int(df.isna().sum().sum())}, clases={y.value_counts().sort_index().tolist()}")
    R["dataset"] = dict(n=len(df), n_feat_total=Xall.shape[1], n_feat_sin_long=X.shape[1], faltantes=int(df.isna().sum().sum()))
    lon = Xall["Longitud"].groupby(y).agg(["min", "mean", "max"]).round(0)
    R["longitud"] = {int(k): [float(v) for v in r] for k, r in lon.iterrows()}
    R["acc_solo_longitud"] = float(evaluar(DecisionTreeClassifier(random_state=SEED), Xall[["Longitud"]], y, 3)["acc"].mean())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].boxplot([Xall["Longitud"][y == k] for k in range(1, 11)], tick_labels=range(1, 11), patch_artist=True,
                  boxprops=dict(facecolor="#cfe3f1"), medianprops=dict(color=C["naranja"]))
    ax[0].set(xlabel="Clase", ylabel="Longitud (muestras)", title="La longitud ordena casi por sí sola las clases")
    z = (X - X.mean()) / X.std(); top = X.skew().abs().sort_values(ascending=False).index[:6]
    ax[1].boxplot([z[c].clip(-6, 6) for c in top], tick_labels=[corto(c) for c in top], patch_artist=True, boxprops=dict(facecolor="#f4d9cf"))
    ax[1].tick_params(axis="x", rotation=45, labelsize=7)
    ax[1].set(ylabel="z-score (recortado a ±6)", title="Características más asimétricas (outliers)")
    fig.tight_layout(); fig.savefig(ROOT / "figs/fig01_longitud_distribuciones.png", dpi=160); plt.close(fig)
    cm0 = X.corr(); fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm0.values, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False); nm = [corto(c) for c in X.columns]
    ax.set_xticks(range(len(nm)), nm, rotation=90, fontsize=6); ax.set_yticks(range(len(nm)), nm, fontsize=6)
    fig.colorbar(im, shrink=.8, label="r de Pearson"); ax.set_title("Matriz de correlación inicial (29 características, sin Longitud)")
    fig.tight_layout(); fig.savefig(ROOT / "figs/fig02_correlacion_inicial.png", dpi=160); plt.close(fig)

    # ---------------- Fase 2: MLP con todas las características ----------------
    piloto = dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, learning_rate_init=1e-3)
    cur_p = curvas(X, y, piloto, 600); max_iter, ep_est = elegir_iter(cur_p)
    print(f"max_iter elegido={max_iter} (meseta de validación hacia la época {ep_est})")
    grid = {"mlp__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [1e-4, 1e-2], "mlp__learning_rate_init": [1e-3, 1e-2]}
    if a.rapido:
        grid = {"mlp__hidden_layer_sizes": [(64,), (64, 32)], "mlp__activation": ["relu", "tanh"],
                "mlp__alpha": [1e-2], "mlp__learning_rate_init": [1e-3]}
    gs = GridSearchCV(mlp_pipe({}, max_iter), grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                      n_jobs=-1, refit=False).fit(X, y)
    G = pd.DataFrame(gs.cv_results_)[["param_mlp__hidden_layer_sizes", "param_mlp__activation", "param_mlp__alpha",
                                      "param_mlp__learning_rate_init", "mean_test_score", "std_test_score"]]
    G.columns = ["capas", "activacion", "alpha", "lr", "acc_media", "acc_std"]; G["capas"] = G["capas"].astype(str)
    G = G.sort_values("acc_media", ascending=False).reset_index(drop=True)
    G.to_csv(ROOT / "resultados/rejilla_hiperparametros.csv", index=False)
    bp = {k.replace("mlp__", ""): v for k, v in gs.cv_results_["params"][int(np.argmax(gs.cv_results_["mean_test_score"]))].items()}
    print("Mejores hiperparámetros:", bp, "CV acc =", round(G.acc_media[0], 4))
    tl = lambda d: {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
    R["hiper"] = dict(mejores=tl(bp), max_iter=max_iter, epoca_meseta=ep_est, grid=G.round(4).to_dict("records"), piloto=tl(piloto))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); x = np.arange(1, 601)
    m = cur_p["a_va"].mean(0); s = cur_p["a_va"].std(0)
    ax[0].plot(x, cur_p["a_tr"].mean(0), color=C["azul"], label="Entrenamiento"); ax[0].plot(x, m, color=C["naranja"], label="Validación")
    ax[0].fill_between(x, m - s, m + s, color=C["naranja"], alpha=.15); ax[0].axvline(max_iter, ls="--", color="k", lw=1, label=f"max_iter = {max_iter}")
    ax[0].set(xlabel="Época", ylabel="Exactitud", title="Curva piloto: exactitud (5 pliegues)"); ax[0].legend()
    ax[1].plot(x, cur_p["l_tr"].mean(0), color=C["azul"], label="Entrenamiento"); ax[1].plot(x, cur_p["l_va"].mean(0), color=C["naranja"], label="Validación")
    ax[1].axvline(max_iter, ls="--", color="k", lw=1); ax[1].set(xlabel="Época", ylabel="Log-loss", title="Curva piloto: pérdida"); ax[1].legend()
    fig.tight_layout(); fig.savefig(ROOT / "figs/fig03_curva_piloto.png", dpi=160); plt.close(fig)
    tp = G.head(12).iloc[::-1]; fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([f"{r.capas} {r.activacion} a={r.alpha:g} lr={r.lr:g}" for r in tp.itertuples()], tp.acc_media * 100,
            xerr=tp.acc_std * 100, color=C["verde"], ecolor=C["gris"])
    ax.set(xlabel="Exactitud media CV-5 (%)", title="Rejilla de hiperparámetros: mejores configuraciones"); ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(max(0, tp.acc_media.min() * 100 - 8), 100); fig.tight_layout(); fig.savefig(ROOT / "figs/fig04_rejilla.png", dpi=160); plt.close(fig)
    P = dict(bp)

    # ---------------- Fase 3: correlación y reducción ----------------
    keep, quitadas = reducir_corr(X); Xr = X[keep]
    pares = [(corto(X.columns[i]), corto(X.columns[j]), round(float(cm0.values[i, j]), 3))
             for i in range(len(X.columns)) for j in range(i) if abs(cm0.values[i, j]) > THR]
    R["n_conservadas_spearman"] = int(len(X.columns) - len(reducir_corr(X.rank())[1]))
    print(f"Pares |r|>{THR}: {len(pares)}; eliminadas {len(quitadas)}; conservadas {len(keep)}")
    R["corr"] = dict(umbral=THR, n_pares=len(pares), pares=pares, conservadas=keep,
                     eliminadas=[dict(quitada=q, par=o, r=round(r, 3)) for q, o, r in quitadas],
                     nombres=X.columns.tolist(), matriz=cm0.round(2).values.tolist())
    pd.DataFrame(quitadas, columns=["eliminada", "redundante_con", "r"]).to_csv(ROOT / "resultados/caracteristicas_eliminadas.csv", index=False)
    pd.concat([Xr, y], axis=1).to_csv(ROOT / "data/dataset_reducido.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2)); cr = Xr.corr()
    for axx, M, tt in ((ax[0], cm0, f"Antes: {X.shape[1]} características"), (ax[1], cr, f"Después: {len(keep)} características")):
        im = axx.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1); axx.grid(False); n = [corto(c) for c in M.columns]
        axx.set_xticks(range(len(n)), n, rotation=90, fontsize=5.5); axx.set_yticks(range(len(n)), n, fontsize=5.5); axx.set_title(tt)
    fig.colorbar(im, ax=ax, shrink=.7, label="r"); fig.savefig(ROOT / "figs/fig05_correlacion_antes_despues.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---------------- Fases 2 y 4: evaluación con los mismos hiperparámetros ----------------
    art_keep = [c for c in Xall.columns if c not in ART_ELIMINADAS and c != "Longitud"]
    k2, k4 = "MLP Fase 2: sin Longitud, 29 car.", f"MLP Fase 4: reducido, {len(keep)} car."
    exps = [("Árbol (artículo): con Longitud, 30 car.", DecisionTreeClassifier(random_state=SEED), Xall),
            ("Árbol: sin Longitud, 29 car.", DecisionTreeClassifier(random_state=SEED), X),
            ("Árbol: sin Longitud, reducido", DecisionTreeClassifier(random_state=SEED), Xr),
            ("MLP: con Longitud, 30 car.", mlp_pipe(P, max_iter), Xall),
            (k2, mlp_pipe(P, max_iter), X), (k4, mlp_pipe(P, max_iter), Xr),
            ("MLP: 18 car. del artículo, sin Longitud", mlp_pipe(P, max_iter), Xall[art_keep])]
    E = {}; rows = []
    for nm, mod, XX in exps:
        r = evaluar(mod, XX, y, reps); E[nm] = r
        rows.append(dict(modelo=nm, n_caracteristicas=XX.shape[1], accuracy=r["acc"].mean(), std=r["acc"].std(),
                         precision_macro=r["prec"], recall_macro=r["rec"], f1_macro=r["f1"]))
        print(f"{nm:45s} acc={r['acc'].mean()*100:5.1f}±{r['acc'].std()*100:4.1f}  f1={r['f1']:.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "resultados/comparacion_modelos.csv", index=False)
    d = E[k4]["acc_rep"] - E[k2]["acc_rep"]
    try: pw = float(wilcoxon(d).pvalue)
    except ValueError: pw = 1.0
    R["comparacion"] = [{k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in r.items()} for r in rows]
    R["f2_vs_f4"] = dict(dif_media=float(d.mean()), p_wilcoxon=pw, reps=reps)
    R["recall_clase"] = {"articulo": ART_RECALL, "arbol_con_long": np.diag(E[exps[0][0]]["cm"]).round(1).tolist(),
                         "mlp_f2": np.diag(E[k2]["cm"]).round(1).tolist(), "mlp_f4": np.diag(E[k4]["cm"]).round(1).tolist()}
    R["cm"] = dict(f2=E[k2]["cm"].round(1).tolist(), f4=E[k4]["cm"].round(1).tolist(), clases=CLASES)
    pd.DataFrame(R["recall_clase"], index=CLASES).to_csv(ROOT / "resultados/recall_por_clase.csv")

    ep = int(max_iter * 1.5); cf = {"Fase 2 (29 car.)": curvas(X, y, P, ep), f"Fase 4 ({len(keep)} car.)": curvas(Xr, y, P, ep)}
    R["curvas"] = {k: {kk: v[kk].mean(0)[::5].round(4).tolist() for kk in v} for k, v in cf.items()}; R["curvas_paso"] = 5
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2)); x = np.arange(1, ep + 1)
    for i, (k, v) in enumerate(cf.items()):
        for col, (key, ttl, yl) in enumerate((("a", "Exactitud", "Exactitud"), ("l", "Pérdida (log-loss)", "Log-loss"))):
            axx = ax[i, col]; axx.plot(x, v[key + "_tr"].mean(0), color=C["azul"], label="Entrenamiento")
            mu, sd = v[key + "_va"].mean(0), v[key + "_va"].std(0); axx.plot(x, mu, color=C["naranja"], label="Validación")
            axx.fill_between(x, mu - sd, mu + sd, color=C["naranja"], alpha=.15); axx.axvline(max_iter, ls="--", color="k", lw=1)
            axx.set(title=f"{k}: {ttl}", xlabel="Época", ylabel=yl)
            if i == 0 and col == 0: axx.legend()
    fig.tight_layout(); fig.savefig(ROOT / "figs/fig06_curvas_fase2_fase4.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for axx, (k, tt) in zip(ax, ((k2, "MLP Fase 2: 29 características"), (k4, f"MLP Fase 4: {len(keep)} características"))):
        im = heat(axx, E[k]["cm"], [c.split(" ", 1)[1] for c in CLASES], CLASES, vmax=100)
        axx.set_title(f"{tt}\nexactitud {E[k]['acc'].mean()*100:.1f} %"); axx.set_xlabel("Clase predicha"); axx.set_ylabel("Clase real")
    fig.colorbar(im, ax=ax, shrink=.75, label="% de la clase real"); fig.savefig(ROOT / "figs/fig07_matrices_confusion.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.6, 4.2)); cols = [C["gris"], C["gris"], C["gris"], C["ambar"], C["azul"], C["verde"], C["naranja"]]
    ax.barh([r["modelo"] for r in rows][::-1], [r["accuracy"] * 100 for r in rows][::-1],
            xerr=[r["std"] * 100 for r in rows][::-1], color=cols[::-1], ecolor="k", capsize=2)
    ax.axvline(83.4, ls="--", color=C["naranja"], lw=1)
    for i, r in enumerate(rows[::-1]): ax.text(r["accuracy"] * 100 + r["std"] * 100 + .8, i, f"{r['accuracy']*100:.1f}", va="center", fontsize=8)
    ax.set(xlim=(0, 108), xlabel=f"Exactitud (%), CV 5 pliegues × {reps} repeticiones (línea: 83.4 % del artículo)", title="Comparación de modelos y conjuntos de características")
    ax.tick_params(axis="y", labelsize=8); fig.tight_layout(); fig.savefig(ROOT / "figs/fig08_comparacion.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.8)); w = .2; xs = np.arange(10)
    for i, (k, lab, col) in enumerate((("articulo", "Artículo (árbol, con Longitud)", C["gris"]), ("arbol_con_long", "Árbol replicado (con Longitud)", C["ambar"]),
                                       ("mlp_f2", "MLP Fase 2", C["azul"]), ("mlp_f4", "MLP Fase 4", C["verde"]))):
        ax.bar(xs + (i - 1.5) * w, R["recall_clase"][k], w, label=lab, color=col)
    ax.set_xticks(xs, CLASES, rotation=30, ha="right", fontsize=8); ax.set(ylabel="Recall por clase (%)", title="Desempeño por patrón")
    ax.legend(fontsize=7, ncol=4, loc="lower center"); fig.tight_layout(); fig.savefig(ROOT / "figs/fig09_recall_por_clase.png", dpi=160); plt.close(fig)

    # ---------------- Separabilidad: PCA + K-means (conexión con la Actividad 5) ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4)); R["pca"] = {}; R["cluster"] = {}
    for axx, (nm, XX) in zip(ax, (("29 características", X), (f"{len(keep)} características", Xr))):
        Z = StandardScaler().fit_transform(XX); P2 = PCA(2, random_state=SEED).fit(Z); Q = P2.transform(Z)
        for k in range(1, 11): axx.scatter(Q[y == k, 0], Q[y == k, 1], s=14, color=plt.cm.tab10(k - 1), label=CLASES[k - 1], alpha=.85)
        axx.set(title=f"PCA – {nm} (var. {P2.explained_variance_ratio_[:2].sum()*100:.0f} %)", xlabel="PC1", ylabel="PC2")
        km = KMeans(10, n_init=10, random_state=SEED).fit(Z)
        R["cluster"][nm] = dict(silueta_clases=float(silhouette_score(Z, y)), ari=float(adjusted_rand_score(y, km.labels_)), nmi=float(normalized_mutual_info_score(y, km.labels_)))
        R["pca"][nm] = dict(x=Q[:, 0].round(3).tolist(), y=Q[:, 1].round(3).tolist(), var=P2.explained_variance_ratio_[:2].round(3).tolist())
    R["pca"]["clase"] = y.tolist(); ax[1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1, .5))
    fig.tight_layout(); fig.savefig(ROOT / "figs/fig10_pca_clases.png", dpi=160); plt.close(fig)
    R["tiempo_s"] = round(time.time() - t0, 1)
    (ROOT / "docs/data/resultados.json").write_text(json.dumps(R, ensure_ascii=False))
    print(f"Listo en {R['tiempo_s']} s. Resultados en resultados/, figs/ y docs/data/resultados.json")


if __name__ == "__main__":
    main()
