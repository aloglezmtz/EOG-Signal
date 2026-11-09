#!/usr/bin/env python3
"""Genera reporte/Reporte_Proyecto_Final_EOG_MLP.pdf a partir de resultados/ y figs/ (ejecutar antes eog_mlp.py)."""
import json
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).parent
R = json.loads((ROOT / "docs/data/resultados.json").read_text())
OUT = ROOT / "reporte"; OUT.mkdir(exist_ok=True)
NAVY, TEAL = colors.HexColor("#1F3A5F"), colors.HexColor("#2A9D8F")
AUTORA, CODIGO = "Alondra Goretti González Martínez y Emiliano Barraza Cervantes", "218534681 · 219425754"

body = ParagraphStyle("b", fontName="Times-Roman", fontSize=11, leading=15.5, alignment=TA_JUSTIFY, firstLineIndent=18, spaceAfter=5)
bodyni = ParagraphStyle("bn", parent=body, firstLineIndent=0)
h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=12, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=8, spaceAfter=3)
cap = ParagraphStyle("cap", fontName="Times-Bold", fontSize=10, leading=12, spaceBefore=6)
capt = ParagraphStyle("capt", fontName="Times-Italic", fontSize=10, leading=12, spaceAfter=3)
nota = ParagraphStyle("nota", fontName="Times-Roman", fontSize=8.5, leading=10.5, spaceAfter=8)
cell = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.5, leading=10)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")
bul = ParagraphStyle("bul", parent=body, firstLineIndent=0, leftIndent=16, bulletIndent=4, spaceAfter=2)
code = ParagraphStyle("code", fontName="Courier", fontSize=7, leading=8.4)

story, nfig, ntab = [], [0], [0]
def P(t, s=body): story.append(Paragraph(t, s))
def H1(t): story.append(Paragraph(t, h1))
def H2(t): story.append(Paragraph(t, h2))
def B(items):
    for t in items: story.append(Paragraph(t, bul, bulletText="•"))
def FIG(img, titulo, nt, w=15.5):
    nfig[0] += 1; im = Image(str(ROOT / "figs" / img)); r = im.imageHeight / im.imageWidth; im.drawWidth, im.drawHeight = w * cm, w * cm * r
    story.append(KeepTogether([Paragraph(f"Figura {nfig[0]}", cap), Paragraph(titulo, capt), im, Paragraph(f"<i>Nota.</i> {nt}", nota)]))
def TAB(titulo, rows, widths, nt, hdr=1, align_first_left=True):
    ntab[0] += 1
    data = [[Paragraph(str(c), cellb if i < hdr else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=hdr)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, hdr - 1), colors.HexColor("#DCE8F2")), ("LINEABOVE", (0, 0), (-1, 0), 1, NAVY),
                           ("LINEBELOW", (0, hdr - 1), (-1, hdr - 1), .6, NAVY), ("LINEBELOW", (0, -1), (-1, -1), 1, NAVY),
                           ("ROWBACKGROUNDS", (0, hdr), (-1, -1), [colors.white, colors.HexColor("#F6F9FC")]),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    story.append(KeepTogether([Paragraph(f"Tabla {ntab[0]}", cap), Paragraph(titulo, capt), t, Paragraph(f"<i>Nota.</i> {nt}", nota)]))

class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=letter, leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.3 * cm, bottomMargin=2.2 * cm,
                         title="Proyecto Final - Agrupamiento de señales EOG Blink-to-Speak con MLP", author=AUTORA)
        self.addPageTemplates([PageTemplate("p", [Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")], onPage=self.deco)])
    def deco(self, c, d):
        if d.page > 1:
            c.setFont("Times-Italic", 8); c.setFillColor(colors.grey)
            c.drawString(2.5 * cm, letter[1] - 1.5 * cm, "Proyecto Final · EOG con MLP")
            c.drawRightString(letter[0] - 2.5 * cm, letter[1] - 1.5 * cm, "A. G. González Martínez (218534681) · E. Barraza Cervantes (219425754)")
            c.setStrokeColor(NAVY); c.line(2.5 * cm, letter[1] - 1.65 * cm, letter[0] - 2.5 * cm, letter[1] - 1.65 * cm)
            c.setFont("Times-Roman", 9); c.drawCentredString(letter[0] / 2, 1.3 * cm, str(d.page))
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            self.notify("TOCEntry", (0 if f.style.name == "h1" else 1, f.getPlainText(), self.page))

# ------------- datos -------------
cmp_ = {r["modelo"]: r for r in R["comparacion"]}; M = list(cmp_.values())
a_arbL, a_arbS, a_arbR, a_mlpL, a_f2, a_f4, a_a19 = [m["accuracy"] * 100 for m in M]; a_art = a_arbL
s_f2, s_f4 = M[4]["std"] * 100, M[5]["std"] * 100
hp, mi = R["hiper"]["mejores"], R["hiper"]["max_iter"]
nk = len(R["corr"]["conservadas"]); ne = len(R["corr"]["eliminadas"])
mine = {e["quitada"] for e in R["corr"]["eliminadas"]}
art = {"SC_Std_Derecha", "SC_Std_Izquierda", "SC_Std_Horizontales", "SC_Multiplicacion_Std", "SC_Suma_Std", "SC_Multiplicacion_RMS",
       "SC_Multiplicacion_Varianza", "SC_Suma_Varianza", "SC_DifMaxMin_Derecha", "SC_DifMaxMin_Izquierda", "SC_DifMaxMin_Horizontales"}
solapa = mine & art; solo_mias = sorted(mine - art); solo_art = sorted(art - mine)
dif, pw = R["f2_vs_f4"]["dif_media"] * 100, R["f2_vs_f4"]["p_wilcoxon"]
rc = R["recall_clase"]; CL = R["cm"]["clases"]; cl29, cl18 = R["cluster"]["29 características"], R["cluster"][f"{nk} características"]
f1 = lambda v: f"{v:.1f}"
lon = R["longitud"]

# ------------- portada -------------
story += [Spacer(1, 2.2 * cm), Paragraph("PROYECTO FINAL", ParagraphStyle("t0", fontName="Times-Bold", fontSize=11, textColor=TEAL, alignment=TA_CENTER, spaceAfter=10)),
          Paragraph("Agrupamiento de señales EOG del lenguaje Blink-to-Speak con una red neuronal MLP",
                    ParagraphStyle("t1", fontName="Times-Bold", fontSize=24, leading=29, textColor=NAVY, alignment=TA_CENTER, spaceAfter=10)),
          Paragraph("Diez patrones sin la característica «longitud», estandarización z-score, reducción por correlación (|r| &gt; 0.8) y comparación con el árbol de decisión del artículo",
                    ParagraphStyle("t2", fontName="Times-Italic", fontSize=12.5, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#444"))),
          Spacer(1, 1.2 * cm), Paragraph("Reporte técnico · Elaborado con Python 3 (pandas, NumPy, scikit-learn, matplotlib)", ParagraphStyle("t3", fontName="Times-Roman", fontSize=10, alignment=TA_CENTER, spaceAfter=18))]
ficha = [["Integrantes", "Alondra Goretti González Martínez (código 218534681)<br/>Emiliano Barraza Cervantes (código 219425754)"], ["Profesor", "Emilio Barajas González"], ["Actividad", "Proyecto Final: MLP sobre señales EOG (100 puntos)"],
         ["Fecha de entrega", "23 de noviembre de 2026"], ["Conjunto de datos", "EOG Blink-to-Speak (Padilla-Becerra et al., 2024) · 200 registros · 30 características + clase"],
         ["Entregables", "Reporte PDF · Dataset reducido (CSV) · Código Python · Página GitHub Pages · Video"]]
t = Table([[Paragraph(a, ParagraphStyle("fa", fontName="Times-Bold", fontSize=10.5, textColor=NAVY)), Paragraph(b, ParagraphStyle("fb", fontName="Times-Roman", fontSize=10.5, leading=13))] for a, b in ficha], colWidths=[4.2 * cm, 11.8 * cm])
t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), .4, colors.lightgrey), ("LINEABOVE", (0, 0), (-1, 0), 1.2, NAVY), ("LINEBELOW", (0, -1), (-1, -1), 1.2, NAVY),
                       ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
story += [t, Spacer(1, 1.2 * cm)]
P(f"<b>Resumen.</b> Se reprodujo la clasificación de los diez patrones oculares del lenguaje Blink-to-Speak registrados con EOG, pero sin usar la longitud de la señal, "
  f"que el árbol de decisión del artículo utiliza como variable principal. Con un MLP (capas {tuple(hp['hidden_layer_sizes'])}, activación {hp['activation']}, "
  f"{mi} iteraciones) y validación cruzada de 5 pliegues repetida 10 veces, la exactitud con las 29 características restantes fue de <b>{a_f2:.1f} %</b> "
  f"(azar: 10 %); al eliminar por correlación |r| &gt; 0.8 y quedarse con {nk} características bajó a <b>{a_f4:.1f} %</b> (diferencia de {dif:.1f} puntos, p = {pw:.3f}). "
  f"Al devolver la longitud, el MLP sube a {a_mlpL:.1f} % y el árbol a {a_arbL:.1f} %, cifra casi idéntica al 83.4 % publicado. Se concluye que los diez patrones solo son parcialmente "
  "identificables sin la longitud y que buena parte del desempeño reportado proviene de la duración de cada registro, que a su vez depende del protocolo de adquisición.", bodyni)
story.append(PageBreak())
toc = TableOfContents(); toc.levelStyles = [ParagraphStyle("t1", fontName="Times-Bold", fontSize=10.5, leading=15, leftIndent=0), ParagraphStyle("t2", fontName="Times-Roman", fontSize=10, leading=13, leftIndent=16)]
story += [Paragraph("Contenido", ParagraphStyle("hc", parent=h1, spaceBefore=0)), toc, PageBreak()]

# ------------- 1 Introducción -------------
H1("1. Introducción y objetivos")
P("Blink to Speak (BTS) es un lenguaje ocular que permite expresar mensajes mediante combinaciones de parpadeos y movimientos de los ojos. Padilla-Becerra et al. (2024) registraron con electrooculografía (EOG) "
  "diez de esos mensajes en 20 participantes, extrajeron características estadísticas de las señales y entrenaron un árbol de decisión que alcanzó 83.4 % de exactitud. "
  "Sin embargo, la longitud del registro (número de muestras) es la variable de la raíz del árbol y sola ordena buena parte de las clases, de modo que el resultado podría sobrestimar qué tan distinguibles son los patrones en sí.")
P(f"Este proyecto examina esa duda con una red neuronal perceptrón multicapa (MLP). Se pregunta si las diez clases forman grupos identificables <i>sin</i> la longitud, y cómo cambia la respuesta al eliminar características "
  "altamente correlacionadas. El trabajo parte del dataset del artículo y del enfoque de agrupamiento y comparación de la Actividad 5 (K-means sobre Iris): estandarizar, probar varias configuraciones, medir con métricas objetivas y contrastar con las etiquetas reales solo después.")
H2("Objetivo general")
P("Aplicar un MLP a las señales EOG del lenguaje Blink-to-Speak sin la característica «longitud», comparar su desempeño antes y después de reducir la dimensionalidad por correlación y evaluar críticamente si las diez clases originales son identificables.", bodyni)
H2("Objetivos específicos")
B(["Preprocesar el dataset: eliminar «Longitud», estandarizar con z-score y visualizar distribuciones y correlaciones (Fase 1).",
   "Entrenar un MLP con distintos hiperparámetros, validación cruzada, curvas de aprendizaje y métricas de clasificación (Fase 2).",
   "Calcular la matriz de correlación, identificar pares con |r| &gt; 0.8 y eliminar características redundantes (Fase 3).",
   "Repetir el MLP con los mismos hiperparámetros sobre el dataset reducido y compararlo con la Fase 2 (Fase 4).",
   "Contrastar con los resultados del artículo (árbol de decisión), reportar las características que el artículo eliminó y responder las cinco preguntas de análisis crítico (Fase 5)."])

# ------------- 2 Sustento teórico -------------
H1("2. Sustento teórico")
H2("2.1 Electrooculografía y lenguaje Blink-to-Speak")
P("La EOG mide la diferencia de potencial entre la córnea y la retina; los movimientos oculares la modifican y se registran con electrodos de superficie. En el artículo se usaron tres canales (horizontal, derecho e izquierdo) "
  "a 256 Hz con un equipo Natus XLTEK EEG 32, filtrados con un pasa-banda de fase cero de 0.5 a 30 Hz. Cada mensaje BTS combina de uno a cuatro movimientos (Tabla 1) y los participantes los ejecutaron al ritmo de un conteo por segundo.")
TAB("Mensajes seleccionados y número de movimientos oculares", [["Clase", "Mensaje", "Patrón ocular", "Mov."],
    ["1", "Sí", "1 parpadeo", "1"], ["2", "No", "2 parpadeos", "2"], ["3", "Quiero dormir", "4 parpadeos", "4"], ["4", "Agua", "1 izquierda, 1 derecha", "2"],
    ["5", "Baño", "1 arriba", "1"], ["6", "Quiero comer", "1 arriba, 1 abajo", "2"], ["7", "No estoy bien", "1 izquierda, 1 derecha, 1 parpadeo", "3"],
    ["8", "Falta de aire", "Mirar a la izquierda sostenido", "1"], ["9", "Dolor", "1 guiño, 1 parpadeo", "2"], ["10", "Peligro", "Cierre largo y parpadeo frenético", "2"]],
    [1.5, 3.6, 8, 1.6], "Elaborada a partir de las Figuras 2 y 3 y la Tabla 3 de la presentación del artículo.")
H2("2.2 Perceptrón multicapa (MLP)")
P("Un MLP transforma el vector de entrada con capas de neuronas que aplican una combinación lineal y una función de activación no lineal (ReLU o tanh); la capa de salida usa softmax para producir una probabilidad por clase. "
  "Los pesos se ajustan minimizando la entropía cruzada con descenso de gradiente estocástico (aquí, Adam). Sus hiperparámetros principales son la arquitectura, la activación, el factor de regularización L2 (alpha), la tasa de aprendizaje inicial y el número máximo de iteraciones (épocas). "
  "Como es sensible a la escala de las variables, se estandariza con z = (x − media) / desviación estándar, calculando la media y la desviación únicamente con los datos de entrenamiento de cada pliegue para no filtrar información del conjunto de prueba.")
H2("2.3 Validación cruzada, correlación y métricas")
P("En la validación cruzada estratificada de k pliegues cada pliegue conserva la proporción de clases; se entrena con k−1 pliegues y se prueba con el restante. Se usó k = 5 repetida 10 veces, igual que el artículo. "
  "La correlación de Pearson r mide la relación lineal entre dos características; si |r| &gt; 0.8 se considera que una es redundante respecto de la otra. Las métricas de clasificación son la exactitud (accuracy), la precisión, el recall y la F1 macro, y la matriz de confusión por clase.")

# ------------- 3 Metodología -------------
H1("3. Metodología")
P("Todo el flujo está en el script <font face='Courier'>eog_mlp.py</font> (Anexo A) y se ejecuta con <font face='Courier'>python eog_mlp.py</font>. Las semillas están fijadas (42) para que los resultados sean reproducibles.")
H2("3.1 Datos")
P(f"El archivo original del salón se entregó como un PDF exportado de Google Sheets; sus 28 páginas se reconstruyeron con pdfplumber en un CSV de {R['dataset']['n']} filas y {R['dataset']['n_feat_total']} características más la clase "
  f"(20 registros por clase, sin datos faltantes). Como verificación de la reconstrucción, un árbol de decisión con todas las características (incluida la longitud) y el mismo protocolo del artículo obtuvo {a_art:.1f} % frente al 83.4 % publicado.")
H2("3.2 Fases del proyecto")
TAB("Resumen de las fases ejecutadas", [["Fase", "Procedimiento"],
    ["1. Preprocesamiento", "Se elimina «Longitud» (29 características). Estandarización z-score dentro de cada pliegue. Gráficas de distribución y correlación inicial."],
    ["2. MLP, todas las características", f"Curva piloto para elegir las iteraciones ({mi}). Rejilla de 32 configuraciones con CV-5. Evaluación final con CV-5 × 10 y métricas."],
    ["3. Correlación", f"Matriz de Pearson, pares con |r| &gt; 0.8 ({R['corr']['n_pares']} pares), eliminación iterativa de la característica más redundante ({ne} eliminadas, {nk} conservadas)."],
    ["4. MLP, dataset reducido", "Mismos hiperparámetros e iteraciones de la Fase 2. Curvas de aprendizaje y comparación con la Fase 2 (prueba de Wilcoxon por repetición)."],
    ["5. Análisis crítico", "Comparación con el árbol del artículo, características eliminadas por el artículo, PCA y K-means (k = 10) como evidencia de separabilidad, y respuestas a las 5 preguntas."]],
    [4.4, 11.6], "CV-5 × 10 = validación cruzada de 5 pliegues repetida 10 veces con semillas distintas.")
H2("3.3 Criterios de decisión")
B([f"<b>Iteraciones.</b> Con una configuración piloto (64, 32), ReLU, alpha 1e-4, tasa 1e-3 se registró por época la exactitud y la pérdida en entrenamiento y validación. Se tomó la primera época en que la exactitud de validación suavizada queda a 0.5 puntos de su máximo ({R['hiper']['epoca_meseta']}) y se redondeó hacia arriba a {mi}.",
   "<b>Rejilla.</b> Arquitecturas (32), (64), (64, 32) y (128, 64); activaciones ReLU y tanh; alpha 1e-4 y 1e-2; tasas 1e-3 y 1e-2. Se eligió la mejor exactitud media de CV-5 y esos hiperparámetros se congelaron para la Fase 4.",
   "<b>Eliminación por correlación.</b> Mientras exista un par con |r| &gt; 0.8, se toma el par con mayor |r| y se descarta la característica con mayor correlación media con las demás (la más redundante).",
   "<b>Comparación con el artículo.</b> Se replicó su protocolo (CV-5 × 10) y se reportan las mismas métricas, además de exactitud por clase."])

# ------------- 4 Resultados -------------
H1("4. Resultados")
H2("4.1 Fase 1: preprocesamiento y exploración")
P(f"La longitud separa las clases por bloques (Figura 1): la clase 3 (cuatro parpadeos) ocupa {lon['3'][0]:.0f}–{lon['3'][2]:.0f} muestras y la 10 (peligro) {lon['10'][0]:.0f}–{lon['10'][2]:.0f}, mientras las demás se mueven entre {lon['5'][0]:.0f} y {lon['9'][2]:.0f}. "
  f"Un árbol que usa solo la longitud alcanza {R['acc_solo_longitud']*100:.1f} % de exactitud con diez clases, lo que justifica retirarla. Algunas características tienen outliers extremos (productos de canales con valores de hasta 10<super>23</super>), por lo que el z-score no las vuelve comparables en forma o en cola.")
FIG("fig01_longitud_distribuciones.png", "Longitud por clase y características más asimétricas", "Izquierda: diagramas de caja de la longitud por clase. Derecha: seis características con mayor asimetría tras el z-score (valores recortados a ±6 para poder verlas).")
FIG("fig02_correlacion_inicial.png", "Matriz de correlación inicial de las 29 características", "Correlación de Pearson. Los bloques rojos corresponden sobre todo a varianza, desviación estándar y diferencia máxima–mínima de un mismo canal.", w=11)
H2("4.2 Fase 2: MLP con las 29 características")
FIG("fig03_curva_piloto.png", "Curva de aprendizaje piloto para definir las iteraciones", f"Promedio de 5 pliegues. La exactitud de validación se estabiliza hacia la época {R['hiper']['epoca_meseta']}; se fijó max_iter = {mi}. La brecha entre entrenamiento (cerca de 95 %) y validación (cerca de 55 %) indica sobreajuste por pocas muestras.")
g = pd.DataFrame(R["hiper"]["grid"]).head(8)
TAB("Mejores configuraciones de la rejilla de hiperparámetros", [["Capas ocultas", "Activación", "alpha", "Tasa inicial", "Exactitud CV-5 (%)", "Desv. est."]] +
    [[r.capas, r.activacion, f"{r.alpha:g}", f"{r.lr:g}", f1(r.acc_media * 100), f1(r.acc_std * 100)] for r in g.itertuples()], [3, 2.4, 2, 2.4, 3.4, 2.2],
    f"Ocho de 32 configuraciones, ordenadas por exactitud. Se eligió {tuple(hp['hidden_layer_sizes'])}, {hp['activation']}, alpha {hp['alpha']:g}, tasa {hp['learning_rate_init']:g}. alpha 1e-4 y 1e-2 dieron resultados idénticos: con z-score y 160 muestras de entrenamiento la penalización L2 es despreciable en este rango.")
FIG("fig04_rejilla.png", "Exactitud de las doce mejores configuraciones", "Barras de error: desviación estándar entre pliegues. Las diferencias entre las primeras configuraciones son menores que la variabilidad entre pliegues.", w=12.5)
H2("4.3 Fase 3: correlación y reducción de características")
P(f"Se encontraron {R['corr']['n_pares']} pares con |r| &gt; 0.8. La eliminación iterativa descartó {ne} características y dejó {nk} (Tabla 4). Con correlación de rangos (Spearman) se habrían conservado {R['n_conservadas_spearman']}, un resultado parecido, así que la redundancia no se debe solo a los outliers.")
TAB("Características eliminadas por correlación", [["Eliminada", "Redundante con", "|r|"]] + [[e["quitada"], e["par"], f"{e['r']:.3f}"] for e in R["corr"]["eliminadas"]], [6.2, 6.2, 2],
    "Se elimina en cada paso la característica del par con mayor correlación media con el resto.")
FIG("fig05_correlacion_antes_despues.png", "Matriz de correlación antes y después de la reducción", f"Tras la reducción ya no queda ningún par con |r| &gt; 0.8; no fue necesario volver a estandarizar porque el z-score se calcula dentro de cada pliegue sobre las columnas restantes.")
H2("4.4 Fase 4: MLP con el dataset reducido")
FIG("fig06_curvas_fase2_fase4.png", "Curvas de aprendizaje con los mismos hiperparámetros: Fase 2 y Fase 4", f"Línea discontinua: max_iter = {mi}. La exactitud de validación se mantiene plana desde la época ~50 en ambas fases, pero la pérdida de validación empieza a subir hacia la época 65 (sobreajuste); max_iter = 150 es una elección conservadora para la exactitud, y un paro temprano cerca de 65–100 épocas sería una mejora posible.")
TAB("Comparación de modelos y conjuntos de características", [["Modelo", "Car.", "Exactitud (%)", "Precisión macro", "Recall macro", "F1 macro"]] +
    [[m["modelo"], m["n_caracteristicas"], f"{m['accuracy']*100:.1f} ± {m['std']*100:.1f}", f"{m['precision_macro']:.3f}", f"{m['recall_macro']:.3f}", f"{m['f1_macro']:.3f}"] for m in M],
    [6.3, 1.1, 2.7, 2.2, 2, 1.7], f"CV-5 × 10 (± = desviación estándar entre 50 pliegues). Artículo (árbol con longitud): 83.4 %. MLP Fase 4 frente a Fase 2: {dif:.1f} puntos, prueba de Wilcoxon sobre las 10 repeticiones p = {pw:.3f}.")
FIG("fig08_comparacion.png", "Exactitud por modelo y conjunto de características", "La línea punteada marca el 83.4 % reportado en el artículo.", w=13.5)
FIG("fig07_matrices_confusion.png", "Matrices de confusión promedio de la Fase 2 y la Fase 4", "Porcentaje de cada clase real asignado a cada clase predicha, sumando las 10 repeticiones de CV-5.")
TAB("Recall por clase (%)", [["Clase", "Artículo (árbol, con longitud)", "Árbol replicado (con longitud)", "MLP Fase 2", "MLP Fase 4"]] +
    [[CL[i], f1(rc["articulo"][i]), f1(rc["arbol_con_long"][i]), f1(rc["mlp_f2"][i]), f1(rc["mlp_f4"][i])] for i in range(10)], [3.4, 3.6, 3.6, 2.6, 2.6],
    "Los valores del artículo corresponden a la diagonal de su matriz de confusión (Fig. 4 del artículo).")
FIG("fig09_recall_por_clase.png", "Recall por patrón en los cuatro modelos", "Las clases 3 y 10, perfectas en el artículo, son las que más caen sin la longitud.")
H2("4.5 Fase 5: evidencia adicional de separabilidad")
P(f"Para contrastar la lectura con un método no supervisado, como en la Actividad 5, se agruparon los registros con K-means (k = 10) sobre las variables estandarizadas. El índice de Rand ajustado frente a las clases fue {cl29['ari']:.2f} con 29 características y {cl18['ari']:.2f} con {nk}; "
  f"la información mutua normalizada fue {cl29['nmi']:.2f} y {cl18['nmi']:.2f}. La silueta calculada con las etiquetas reales es negativa ({cl29['silueta_clases']:.2f} y {cl18['silueta_clases']:.2f}), lo que indica que las clases no forman nubes compactas y separadas en este espacio.")
FIG("fig10_pca_clases.png", "Proyección PCA de los registros coloreados por clase", "Las dos primeras componentes explican poca varianza y los outliers dominan la escala; las clases se traslapan.")
H2("4.6 Características eliminadas en el artículo")
P("El artículo calculó 34 características candidatas (media, varianza, desviación estándar, RMS, número y distancia de picos y diferencia máxima–mínima, en los tres canales por separado y en su suma y producto, más la longitud) y conservó 19 con correlación menor que 0.8; "
  "no enumera las eliminadas, pero la Figura 6 de la presentación señala las que quedaron fuera del conjunto retenido. De las 30 columnas del dataset entregado se descartaron 11:")
B(["Desviación estándar: derecha, izquierda, horizontal, producto y suma (5).", "RMS del producto (1).", "Varianza del producto y de la suma (2).", "Diferencia máxima–mínima: derecha, izquierda y horizontal (3)."])
P(f"Se conservaron: longitud, varianza de los tres canales por separado, medias (5), número de picos (5) y distancia entre picos (5). El criterio aplicado eliminó también {ne} características y coincidió en {len(solapa)} de las 11 del artículo; "
  f"difiere en que descartó la varianza de los canales derecho e izquierdo ({', '.join(c.replace('SC_', '') for c in solo_mias)}) y conservó la diferencia máxima–mínima derecha e izquierda ({', '.join(c.replace('SC_', '') for c in solo_art)}). "
  f"Entrenar el MLP solo con las 18 características del artículo (sin longitud) dio {a_a19:.1f} %, cifra equivalente a la del dataset reducido propio ({a_f4:.1f} %).")

# ------------- 5 Análisis crítico -------------
H1("5. Análisis crítico")
H2("5.1 ¿Los 10 patrones del artículo son identificables sin usar longitud?")
P(f"Solo en parte. El mejor MLP alcanza {a_f2:.1f} % (± {s_f2:.1f}) con diez clases, claramente por encima del azar (10 %) pero lejos del 83.4 % del artículo. Seis patrones (agua, baño, comer, no estoy bien, falta de aire y dolor) se reconocen entre 60 % y 73 %, "
  f"mientras que «sí» ({rc['mlp_f2'][0]:.0f} %), «no» ({rc['mlp_f2'][1]:.0f} %), «quiero dormir» ({rc['mlp_f2'][2]:.0f} %) y «peligro» ({rc['mlp_f2'][9]:.0f} %) quedan muy por debajo. K-means y PCA tampoco muestran grupos naturales. "
  "En un sistema de comunicación para personas con discapacidad, una exactitud cercana a 54 % no es suficiente para mensajes críticos como «peligro».")
H2("5.2 ¿Qué tanto ayudaron o perjudicaron las características originales?")
P(f"La longitud aporta muchísimo: agregarla sube el MLP de {a_f2:.1f} % a {a_mlpL:.1f} % (+{a_mlpL-a_f2:.1f} puntos) y el árbol de {a_arbS:.1f} % a {a_arbL:.1f} % (+{a_arbL-a_arbS:.1f}). Sobre todo rescata las clases 3 y 10, que el artículo reporta con 100 %. "
  f"Eso no prueba que el patrón ocular sea distinguible, porque la longitud depende de cuánto dura el patrón dentro del protocolo (un parpadeo por segundo): cuatro parpadeos duran más que uno. Entre las demás, las 11 que el artículo descartó sí servían al MLP: con solo las 18 retenidas el resultado baja a {a_a19:.1f} % frente a {a_f2:.1f} % con las 29. "
  "Las medidas de amplitud y dispersión (desviación estándar, diferencia máxima–mínima) parecen aportar información que las de picos y medias no capturan.")
H2("5.3 ¿La eliminación de características correlacionadas mejoró o empeoró la clasificación?")
P(f"Empeoró: de {a_f2:.1f} % a {a_f4:.1f} % ({dif:.1f} puntos, p = {pw:.3f} en la prueba de Wilcoxon sobre 10 repeticiones); el árbol también bajó ({a_arbS:.1f} % a {a_arbR:.1f} %). Una correlación alta lineal no implica información redundante para un clasificador no lineal, "
  "y las variables eliminadas (varianza y desviación estándar) miden amplitud, que puede aportar información complementaria. La reducción sí simplifica el modelo (de 29 a 18 variables) y es coherente con el criterio del artículo, pero aquí tuvo costo en exactitud.")
H2("5.4 ¿Qué limitaciones del artículo se hicieron evidentes?")
B(["<b>Dependencia de la longitud.</b> Es la raíz del árbol y casi separa por sí sola las clases 3 y 10; mide el protocolo más que el movimiento.",
   "<b>Muestra pequeña frente a las variables.</b> 200 registros (20 por clase) y 30 características dejan unas 7 muestras por variable; con 160 de entrenamiento por pliegue el MLP sobreajusta (Figura 3).",
   "<b>Sin identificador de participante.</b> No se puede validar dejando fuera sujetos completos. Con validación aleatoria, registros de la misma persona pueden caer en entrenamiento y prueba, y eso favorece al modelo; el artículo presenta el modelo como generalizable entre usuarios.",
   "<b>Características sin sustento fisiológico.</b> Se calcularon sumas y productos de canales (sus valores llegan a 10<super>23</super>) que el propio artículo reconoce que el modelo no usó, y no se describió su tratamiento de outliers.",
   "<b>Inconsistencias menores.</b> Banda de filtrado de 0.5–35 Hz en la adquisición y 0.5–30 Hz en el acondicionamiento; umbral de correlación de 0.7 que se cambió a 0.8 tras probar con el mismo modelo; un solo algoritmo y sin ajuste de hiperparámetros.",
   "<b>Clases difíciles.</b> «Sí» y «baño» se confunden entre sí (33 % y 32 %) por tener un solo movimiento y poca activación muscular; el artículo lo atribuye a la baja relación señal-ruido."])
H2("5.5 ¿El dataset está bien construido para un problema de clasificación?")
P("Está bien balanceado (20 por clase), sin datos faltantes y con un protocolo de adquisición descrito, pero tiene debilidades para clasificar: (1) la longitud confunde protocolo con patrón; (2) pocas muestras para el número de variables y para entrenar redes neuronales; "
  "(3) variables muy redundantes y con colas pesadas que requieren transformación o selección; (4) falta el identificador de sujeto y no hay repeticiones por sujeto, de modo que no se puede medir la variabilidad intra e interpersonal; "
  "(5) solo se registraron participantes sanos de 18 a 50 años, no la población objetivo. Un dataset mejor registraría varias repeticiones por patrón y participante, normalizaría la duración (por ejemplo, ventanas fijas o segmentación por evento) y publicaría el identificador de sujeto.")
H2("5.6 Limitaciones de este análisis")
P("La rejilla de hiperparámetros se eligió con la misma validación cruzada usada después para reportar, lo que puede ser algo optimista (aunque las repeticiones usan otras particiones); sin identificador de sujeto la validación es aleatoria; "
  "solo se probó un MLP de scikit-learn sin transformaciones de las colas pesadas; y el dataset proviene de un PDF reconstruido, verificado con la réplica del 83.4 % del árbol.")

# ------------- 6 Conclusiones -------------
H1("6. Conclusión general")
P(f"Sin la longitud, el MLP clasifica los diez patrones con {a_f2:.1f} % de exactitud (más de cinco veces el azar) pero muy por debajo del 83.4 % del artículo, y la reducción por correlación empeora el resultado a {a_f4:.1f} %. "
  "La diferencia entre ambos valores se explica en gran parte por la longitud, que codifica la duración del protocolo y no necesariamente el movimiento ocular. Para que el sistema sea utilizable convendría ampliar y normalizar el dataset, "
  "registrar al participante, validar por sujeto y explorar características de forma de onda por evento (amplitud y pendiente de cada parpadeo o mirada) o modelos sobre la señal completa.")

# ------------- Anexos -------------
H1("Anexo A. Código principal")
P("El script completo (<font face='Courier'>eog_mlp.py</font>) está en el repositorio. A continuación, las funciones centrales.")
src = (ROOT / "eog_mlp.py").read_text().split("\n")
def trozo(ini, fin):
    i = next(k for k, l in enumerate(src) if l.startswith(ini)); j = next(k for k in range(i + 1, len(src)) if src[k].startswith(fin))
    return "\n".join(src[i:j]).rstrip()
for ini, fin in (("def mlp_pipe", "def curvas"), ("def elegir_iter", "def heat")):
    story.append(Preformatted(trozo(ini, fin), code)); story.append(Spacer(1, 6))
H1("Anexo B. Reproducción y publicación en GitHub Pages")
P("El repositorio contiene el código, los datos, los resultados, las figuras, este reporte y una página interactiva en <font face='Courier'>docs/</font>. Para ejecutarlo:")
story.append(Preformatted("pip install -r requirements.txt\npython eog_mlp.py            # Fases 1-5: resultados/, figs/, docs/data/resultados.json\npython generar_reporte.py    # regenera este PDF", code))
P("Para publicar: crear el repositorio en GitHub, subir todo, ir a Settings → Pages, elegir «Deploy from a branch», rama main, carpeta /docs. La página corre un MLP en el navegador con el mismo CSV y muestra los resultados de Python precalculados; "
  "Python no se ejecuta en GitHub Pages (solo sirve archivos estáticos), por eso el flujo de GitHub Actions incluido regenera los resultados en cada push.")
H1("Referencias")
for t in ["Padilla-Becerra, M. C., Macias-Castro, D. K., Salido-Ruiz, R. A., Torres-Ramos, S., y Román-Godínez, I. (2024). EOG signal classification based on Blink-to-Speak language. En J. de J. A. Flores Cuautle et al. (Eds.), <i>CNIB 2023, IFMBE Proceedings, 96</i>, 1–9. Springer. https://doi.org/10.1007/978-3-031-46933-6_27",
          "Kingma, D. P., y Ba, J. (2015). Adam: A method for stochastic optimization. <i>ICLR 2015</i>.",
          "Hubert, L., y Arabie, P. (1985). Comparing partitions. <i>Journal of Classification, 2</i>(1), 193–218.",
          "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830.",
          "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. <i>Journal of Computational and Applied Mathematics, 20</i>, 53–65.",
          "González Martínez, A. G. (2026). <i>Actividad 5: Agrupamiento K-means con el conjunto de datos Iris</i> [Reporte técnico]. Universidad de Guadalajara."]:
    P(t, ParagraphStyle("ref", parent=bodyni, leftIndent=18, firstLineIndent=-18, fontSize=10, leading=13))

Doc(str(OUT / "Reporte_Proyecto_Final_EOG_MLP.pdf")).multiBuild(story)
print("PDF generado")
