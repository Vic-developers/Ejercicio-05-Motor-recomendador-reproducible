# Dataset Card · MovieLens Latest Small

## Fuente, versión, fecha y hash

- Fuente: GroupLens Research, University of Minnesota.
- Recurso: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- Nombre de versión: `ml-latest-small` (release `latest`, conjunto vivo, no snapshot archivado).
- Fecha de descarga de la corrida adjunta: 2026-10-22 UTC.
- SHA-256: `reports/data_audit.json` es la fuente de verdad para los bytes ejecutados.
- No se incluye ni redistribuye el ZIP ni sus archivos extraídos.

## Licencia y cita

El README de GroupLens permite uso de investigación bajo condiciones; requiere atribución, no implicar respaldo institucional y no autoriza uso comercial o de ingresos sin permiso. Se respetan además las condiciones de la tarea: los datos no se publican en este repositorio. Consulte los términos vigentes en https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html.

Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4). https://doi.org/10.1145/2827872

## Archivos y campos

- `ratings.csv`: `userId` anónimo, `movieId`, `rating` explícito entre 0.5 y 5.0, `timestamp` Unix.
- `movies.csv`: `movieId`, `title`, `genres` separados por `|`.
- La corrida auditada registra dimensiones, rango temporal, densidad, cohorte y hash en `reports/data_audit.json`.

## Procedimiento de descarga

`uv run python scripts/download_data.py` descarga desde GroupLens, escribe el SHA-256 del ZIP y extrae a `data/raw/`. `.gitignore` mantiene ambas formas fuera de Git. `uv run python scripts/audit_data.py` valida esquema, rating y referencias. El hash debe guardarse con resultados porque `latest` puede cambiar.

## Población observada, calidad y exclusiones

MovieLens Latest Small incluye usuarios que valoraron al menos 20 películas; no es muestra aleatoria de espectadores. No contiene demografía. Ratings reflejan consumo autoseleccionado, y ausencia de una fila no equivale a rechazo. Los timestamps permiten un corte temporal, pero no eliminan sesgo de selección ni garantizan que la simulación represente despliegue. Títulos/géneros y valoraciones pueden contener errores de origen. No inferir preferencias de grupos ni causalidad.
