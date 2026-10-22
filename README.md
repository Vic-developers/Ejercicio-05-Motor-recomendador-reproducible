# INF8239 · Ejercicio 05 · Motor recomendador reproducible

Autor: Edwin Ramón José Nolasco
Asignatura: INF-8239, Unidad 03

## Entrega

Este repositorio integra LAB08 y LAB09: baseline de popularidad suavizada, similitud por géneros, factorización SGD, evaluación temporal, ranking híbrido, perfiles fríos, pruebas, auditoría y este informe. La tabla de resultados se encuentra en [`reports/main_results.csv`](reports/main_results.csv); la tabla de semillas y la comparación Pareto también están en `reports/`. El PDF breve está en [`reports/ejercicio_05_informe.pdf`](reports/ejercicio_05_informe.pdf) y el notebook ejecutado en [`notebooks/ejercicio_05_motor_recomendador.ipynb`](notebooks/ejercicio_05_motor_recomendador.ipynb).

El ZIP de MovieLens y sus CSV no se incluyen en Git. Descárguelos localmente siguiendo el procedimiento de abajo. El proyecto no redistribuye esos datos.

## Reproducibilidad

Requisitos: Python 3.12, `uv` y conexión para la descarga inicial de MovieLens Latest Small.

```powershell
uv sync
uv run pytest -q
uv run python scripts/download_data.py
uv run python scripts/audit_data.py
uv run python scripts/lab08_content.py
uv run python scripts/run_study.py --factors 20 --epochs 12 --seeds 17 42 73 --alphas 0.25 0.75
uv run streamlit run app/streamlit_app.py
```

El primer `uv sync` crea el entorno y fija las dependencias mediante `uv.lock`. MovieLens se descarga desde GroupLens a `data/raw/`; `.gitignore` excluye el ZIP y el directorio extraído. La descarga no cambia el dataset de un experimento ya auditado: el SHA-256 y el resumen usados se guardan en `reports/data_audit.json`. MovieLens Latest Small es una versión viva; para repetir exactamente el experimento, conserve una copia local autorizada y compruebe que su hash coincida.

## Datos y auditoría

Fuente: GroupLens Research, MovieLens Latest Small. `ratings.csv` contiene usuario anónimo, película, rating explícito de 0.5 a 5 y timestamp Unix; `movies.csv` contiene título y géneros. No hay demografía. La auditoría comprueba esquema, rango de ratings y claves foráneas. La densidad mide ratings observados sobre todas las combinaciones usuario-película; un espacio vacío no significa desagrado.

MovieLens Latest Small es un conjunto cambiante para desarrollo. Sus usuarios tienen al menos 20 ratings. No se deben generalizar las métricas offline a la población, a usuarios nuevos ni a una implementación en producción. GroupLens requiere atribución para resultados derivados y prohíbe uso comercial sin autorización; este proyecto no publica los archivos del dataset. Consulte [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) y el README oficial citado allí.

## Métodos y evaluación

- **Popularidad:** promedio bayesiano `n/(n+m)·R + m/(n+m)·C`, donde `R` es la media del ítem, `n` sus ratings, `C` la media global y `m` el umbral de evidencia (percentil 80 de conteos). Se ordena por puntuación suavizada.
- **Contenido:** TF-IDF de géneros y coseno. Expresa coincidencia de metadatos, no preferencia; géneros amplios generan empates y listas repetitivas.
- **Colaborativo:** factorización de matriz con SGD, semilla fija, 20 factores y 12 épocas; los factores latentes no tienen interpretación humana directa.
- **Híbrido:** combinación lineal de puntuación colaborativa normalizada y perfil TF-IDF ponderado por ratings por encima de 2.5. `alpha` de 0.25 y 0.75 pesa contenido y colaboración, respectivamente. El pool híbrido se limita a los 100 candidatos colaborativos mejor puntuados.

El corte ordena por usuario y timestamp y retiene un último evento por usuario; empates de hora se resuelven determinísticamente por `movieId`. Solo se evalúa si usuario e ítem aparecen en entrenamiento: 587 eventos de prueba elegibles en la corrida adjunta. Por ello, el resultado no mide cold start. HitRate@10 cuenta usuarios cuyo único ítem retenido aparece en sus diez recomendaciones; no mide calidad más allá de ese objetivo binario. Cobertura divide ítems recomendados únicos entre los 9,742 ítems del catálogo. RMSE se reporta para modelos que predicen ratings. La comparación Pareto considera hit rate y cobertura mayores, con menor tiempo de ajuste y menor número de parámetros.

La política para usuario sin historial es popularidad global; produce una lista, pero no es personalizada. La cobertura alta puede exponer más catálogo; no demuestra diversidad percibida. Sesgo de selección, popularidad, sobre-especialización, retroalimentación y falta de variables demográficas limitan las conclusiones.

## Resultados adjuntos

Las cifras de `reports/main_results.csv` comparan los cinco métodos bajo el mismo corte temporal y cohorte de 587 usuarios. Popularidad obtuvo HitRate@10 de 0.0017 y 0.13% de cobertura; contenido, 0.0000 y 14.95%; colaborativo, 0.0312 y 7.37%; híbrido α=.25, 0.0290 y 10.38%; híbrido α=.75, 0.0312 y 7.60%. Los modelos factorizados tienen RMSE medio 1.0272 (desviación estándar entre semillas 0.0056). Las cifras corresponden al ZIP cuyo hash aparece en `reports/data_audit.json`. α=.75 iguala el HitRate colaborativo y mejora marginalmente su cobertura; α=.25 sacrifica unos 0.0023 puntos de HitRate para sumar 2.78 puntos porcentuales de cobertura. Ambos pesos están en la frontera Pareto. Se recomienda α=.75 si se prioriza acertar el evento retenido; α=.25 si la prioridad es ampliar exposición. Esta diferencia es descriptiva, no una prueba de significancia.

## Pruebas

```powershell
uv run pytest -q
```

Los tests verifican contratos de datos, exclusión del título consultado, split temporal, predicción finita y definiciones básicas de métricas. No sustituyen la inspección de datos ni validación en usuarios reales.

## Referencias

- GroupLens Research. *MovieLens Latest Small README* (fuente, licencia, descripción y cita bibliográfica): https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4). https://doi.org/10.1145/2827872

## Uso de IA

La declaración completa se encuentra en [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

Se usó OpenAI Codex (GPT-6) como apoyo de programación, revisión del protocolo y redacción. Prompt relevante, resumido fielmente: «Completa el proyecto base de LAB08/LAB09 según el mandato del Ejercicio 05; integra evaluación reproducible de popularidad, contenido, factorización e híbrido con corte temporal, tres semillas, perfiles fríos, comparación Pareto, pruebas, documentación y PDF; publica en el repositorio indicado y no redistribuyas MovieLens». La IA también ayudó a detectar el error de tipo disperso/denso en el coseno y el fallo de índice al ordenar popularidad.

Verificaciones realizadas: auditoría local del dataset y SHA-256; ejecución de LAB08; experimento temporal en tres semillas; pruebas automatizadas y revisión de los reportes y del PDF. Correcciones derivadas: desempate temporal determinista, cohorte elegible explícita, normalización de conversión para coseno, y exclusión de los datos de MovieLens de Git. El estudiante debe revisar y poder explicar cada archivo, cifra, decisión y limitación antes de defenderlo.
