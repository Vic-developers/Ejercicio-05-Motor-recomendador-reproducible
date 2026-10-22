# System Card · Motor recomendador

## Propósito y usuarios

Prototipo académico offline para comparar políticas de recomendación de películas. No está validado para decisiones de producción ni para perfilar personas.

## Catálogo y política de candidatos

Catálogo MovieLens Latest Small, excluyendo elementos que el usuario ya valoró. El colaborativo genera hasta 100 candidatos; el híbrido reordena ese pool. El fallback de usuario sin historial usa popularidad suavizada global y se declara no personalizado.

## Señales y métodos

Ratings explícitos, géneros y timestamps. Baseline de popularidad suavizada; TF-IDF + coseno para contenido; factorización SGD; híbrido con pesos colaborativo `alpha=0.25` y `0.75`. Ver README para ecuaciones y detalles.

## Evaluación

Último evento temporal por usuario, desempate determinista por id de película. 587 de 610 eventos retenidos son elegibles porque el ítem está en el entrenamiento. HitRate@10, cobertura de catálogo, RMSE para modelos de ratings, coste de ajuste y parámetros. Las cifras y semillas están en `reports/`. Evaluación offline, una interacción por usuario, no prueba satisfacción.

## Cold start, cobertura y diversidad

El test excluye usuarios/ítems nuevos no conocidos al modelo. El fallback se comprueba funcionalmente y es global. Cobertura mide catálogo único recomendado, no diversidad intralista ni exposición justa. TF-IDF de géneros puede sobre-especializar y empatar con etiquetas compartidas.

## Riesgos y mitigación

Sesgo de selección y popularidad; retroalimentación; listas repetitivas; factores latentes no explicables; ausencia de datos demográficos; dataset pequeño y cambiante. Mitigación de esta entrega: separación temporal, fallback declarado, cobertura, semillas, comparación Pareto y no publicar ratings. Se requiere evaluación online/usuario y análisis de diversidad antes de desplegar.
