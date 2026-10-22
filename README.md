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
uv run python scripts/run_study.py --configs 8 6 --configs 20 6 --configs 20 12 --seeds 17 42 73 --alphas 0.25 0.75
uv run python scripts/build_deliverables.py
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

El corte ordena por usuario, timestamp y `movieId` para resolver empates, y retiene un último evento por usuario. De 610 eventos retenidos, 587 tienen usuario e ítem conocidos por entrenamiento; los otros 23 se excluyen de HitRate@10 y RMSE. Se mide además el ranking por tamaño de historial: corto (hasta 34 ratings de entrenamiento, n=155) y amplio (desde 167, n=139). HitRate@10 evalúa el único evento retenido; cobertura divide ítems recomendados únicos entre 9,742 películas.

`recommend_for_user` tiene tres rutas comprobadas: usuario conocido recibe híbrido; usuario nuevo con historial recibe contenido; usuario sin historial recibe popularidad suavizada global, excluyendo ítems vistos. El caso sin historial se prueba funcionalmente, no tiene HitRate porque no existe historial para una evaluación temporal válida.

La comparación Pareto prueba tres tamaños de entrenamiento (8 factores/6 épocas, 20/6 y 20/12), tres semillas, dos pesos híbridos, HitRate@10, cobertura, tiempo y parámetros. Los factores 20/12 bajan RMSE y suben HitRate, a costa del tiempo y menor cobertura; el modelo 20/6 da una opción de menor costo y cobertura amplia. Una configuración está dominada si otra no empeora utilidad ni costo y mejora al menos uno. Los resultados son offline; sesgo de selección, popularidad, sobre-especialización, retroalimentación y falta de demografía limitan conclusiones.

## Resultados adjuntos

Los resultados completos por configuración y semilla están en `reports/main_results.csv` y `reports/seed_metrics.csv`. Popularidad suavizada logra HitRate@10=0.0324 con 0.63% de cobertura; contenido, 0.0000 y 14.95%. Para factorización: 8f/6e RMSE=1.108, HR=0.010, cobertura=11.1%, 6.7 s, 82,488 parámetros; 20f/6e RMSE=1.101, HR=0.020, cobertura=18.2%, 6.8 s, 206,220 parámetros; 20f/12e RMSE=1.027, HR=0.031, cobertura=7.4%, 13.6 s.

Decisión: para el híbrido personalizado se elige 20f/6e con alpha=0.75 cuando se busca buen balance entre aciertos, cobertura y costo: frente a 20f/12e reduce el tiempo aproximadamente a la mitad y eleva cobertura de 7.6% a 18.2%, con HitRate@10 de 0.020 frente a 0.031. alpha=0.25 en 20f/6e aumenta cobertura a 21.2% y baja el HitRate a 0.010. La tabla Pareto conserva estas configuraciones como alternativas no dominadas; no hay una ganadora para todos los objetivos. La popularidad global acierta más que el híbrido, pero cubre menos de 1% del catálogo y concentra las listas. Las conclusiones son descriptivas, no pruebas de significancia.

## Pruebas

```powershell
uv run pytest -q
```

Las pruebas cubren contratos de datos, recomendación por contenido, corte temporal, factorización, métricas y las tres rutas de usuario (híbrida conocida, contenido con usuario nuevo e historial, popularidad sin historial). La auditoría descriptiva y el muestreo de errores están en `reports/error_analysis.csv`; el detalle de perfiles y fallback, en `reports/profile_metrics.csv` y `reports/cold_start_validation.json`. No sustituyen validación con usuarios reales.

## Referencias

- GroupLens Research. *MovieLens Latest Small README* (fuente, licencia, descripción y cita bibliográfica): https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4). https://doi.org/10.1145/2827872

## Uso de IA

La declaración completa se encuentra en [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

Se usó OpenAI Codex (GPT-6) como medio de apoyo para detectar y corregir errores y revisar la coherencia de la ejecución. Consultas relevantes: revisar el desempate del corte temporal, errores de tipos en la similitud coseno, índices del baseline de popularidad y consistencia de las métricas. El estudiante conserva la responsabilidad por el código, resultados, referencias y defensa.

Verificaciones realizadas: auditoría local del dataset y SHA-256; ejecución de LAB08; experimento temporal en tres semillas; pruebas automatizadas y revisión de los reportes y del PDF. Correcciones derivadas: desempate temporal determinista, cohorte elegible explícita, normalización de conversión para coseno, y exclusión de los datos de MovieLens de Git. El estudiante debe revisar y poder explicar cada archivo, cifra, decisión y limitación antes de defenderlo.
