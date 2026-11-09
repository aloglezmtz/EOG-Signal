# Guion del video (mínimo 10 minutos)

1. **Presentación (0:00–1:00).** Nombres y códigos de ambos integrantes, materia, objetivo del proyecto y repositorio.
2. **Contexto del artículo (1:00–2:30).** Blink to Speak, 10 mensajes, 20 participantes, EOG de 3 canales, árbol de decisión con 83.4 %. Por qué la longitud es sospechosa (raíz del árbol, Fig. 10 de la presentación).
3. **Datos y Fase 1 (2:30–4:00).** Abrir `docs/index.html`; mostrar el dataset (200 × 30 + clase), boxplot de longitud por clase, quitar Longitud y estandarizar con z-score dentro de cada pliegue.
4. **Fase 2 (4:00–6:00).** Curva piloto para elegir 150 iteraciones, rejilla de 32 configuraciones, mejor MLP (128, 64), tanh, tasa 0.001; resultados: 54.1 %. Matriz de confusión y recall por clase. En el laboratorio de la página, correr el MLP en vivo.
5. **Fase 3 y 4 (6:00–8:00).** Matriz de correlación, 16 pares con |r| > 0.8, 11 características eliminadas, 18 conservadas; mismos hiperparámetros; 45.9 %. Mover el umbral en el laboratorio.
6. **Fase 5 y comparación (8:00–9:30).** Tabla: árbol con longitud 82.5 %, MLP con longitud 76.9 %, MLP sin longitud 54.1 %, reducido 45.9 %, 18 características del artículo 45.6 %. Características que eliminó el artículo.
7. **Preguntas de análisis crítico (9:30–11:30).** Responder las cinco preguntas del PDF (sección 5).
8. **Cierre (11:30–12:00).** Limitaciones del análisis y trabajo futuro: validar por sujeto, normalizar duración, más datos.
