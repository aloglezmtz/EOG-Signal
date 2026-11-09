# Señales EOG Blink-to-Speak con MLP (Proyecto Final)

**Alondra Goretti González Martínez (218534681)** y **Emiliano Barraza Cervantes (219425754)** · Profesor Emilio Barajas González

Clasificación/agrupamiento de los 10 patrones oculares del lenguaje Blink-to-Speak (Padilla-Becerra et al., 2024) con un MLP, **sin la característica `Longitud`**, antes y después de reducir características con |r| > 0.8, y comparación con el árbol de decisión del artículo.

## Resultados (CV 5 pliegues × 10)

| Modelo | Car. | Exactitud |
|---|---|---|
| Árbol con longitud (réplica del artículo, 83.4 %) | 30 | 82.5 % |
| MLP con longitud | 30 | 76.9 % |
| **MLP Fase 2: sin longitud** | 29 | **54.1 %** |
| **MLP Fase 4: reducido por correlación** | 18 | **45.9 %** |

## Estructura

```
eog_mlp.py            # Fases 1-5: preprocesa, entrena, grafica, exporta
generar_reporte.py    # genera reporte/Reporte_Proyecto_Final_EOG_MLP.pdf
data/                 # dataset_eog.csv (completo) y dataset_reducido.csv
resultados/           # CSV de métricas, rejilla, características eliminadas
figs/                 # figuras del reporte
reporte/              # PDF del reporte
docs/                 # sitio de GitHub Pages (index.html + datos + figuras)
.github/workflows/    # pages.yml (publica docs/) y resultados.yml (regenera con Python)
index.html            # redirige a docs/ (por si publicas desde la raíz)
guion_video.md        # guion para el video de 10+ minutos
```

## Ejecutar

```bash
pip install -r requirements.txt
python eog_mlp.py            # completo (~2 min)
python eog_mlp.py --rapido   # prueba rápida
python generar_reporte.py    # regenera el PDF
python -m http.server --directory docs   # ver la página en http://localhost:8000
```

`python eog_mlp.py --csv otro.csv` acepta otro CSV con las mismas columnas (última columna `Clase`).

## Publicar en GitHub Pages

La página está en `docs/` (HTML + CSV + JSON + figuras + PDF) y ya incluye todo lo necesario; no hay que compilar nada.

1. Crea un repositorio nuevo en GitHub (público) y sube **todo** el contenido de esta carpeta, incluidos los archivos ocultos `.github/` y `.nojekyll`:
   ```bash
   git init -b main
   git add -A
   git commit -m "Proyecto final EOG MLP"
   git remote add origin https://github.com/TU-USUARIO/NOMBRE-DEL-REPO.git
   git push -u origin main
   ```
2. En GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**. El flujo `pages.yml` publica `docs/` en cada `push` a `main` (pestaña *Actions* para ver el avance, 1–2 min).
3. La página queda en `https://TU-USUARIO.github.io/NOMBRE-DEL-REPO/`.

Alternativa sin Actions: **Source: Deploy from a branch**, rama `main` y carpeta `/docs` (o `/ (root)`: el `index.html` de la raíz redirige a `docs/`).

Si la página sale en blanco o con 404: confirma que `docs/index.html` esté en el repositorio, que el repo sea público y espera un par de minutos tras el primer despliegue.

GitHub Pages solo sirve archivos estáticos, así que Python no corre ahí: la página muestra los resultados precalculados por `eog_mlp.py` y entrena un MLP en JavaScript en el navegador (laboratorio). El flujo `resultados.yml` vuelve a ejecutar Python en GitHub (*Actions → Regenerar resultados → Run workflow*).

**Video:** al terminar, edita `docs/index.html` (busca `id="vid"`) y pega el enlace de tu video.

## Nota sobre los datos

El dataset se reconstruyó del PDF exportado de Google Sheets (200 registros × 30 características + clase). Como verificación, el árbol de decisión con todas las características reproduce el ~83 % del artículo.
