# System Card · Motor recomendador

## Propósito y usuarios

Prototipo académico offline para comparar políticas de recomendación de películas. No está validado para decisiones de producción ni para perfilar personas.

## Catálogo y política de candidatos

Catálogo MovieLens Latest Small, excluyendo elementos que el usuario ya valoró. El colaborativo genera hasta 100 candidatos; el híbrido reordena ese pool. La función `recommend_for_user` enruta un usuario conocido al híbrido, un usuario nuevo con historial al contenido y un usuario sin historial a popularidad suavizada global. Cada ruta evita recomendar películas ya vistas; el fallback global no es personalizado.

## Señales y métodos

Ratings explícitos, géneros y timestamps. Baseline de popularidad suavizada; TF-IDF + coseno para contenido; factorización SGD; híbrido con pesos colaborativo `alpha=0.25` y `0.75`. Ver README para ecuaciones y detalles.

## Evaluación

Último evento temporal por usuario, desempate determinista por id de película. 587 de 610 eventos retenidos son elegibles porque el usuario y el ítem están en entrenamiento. Se comparan tres configuraciones (8f/6e, 20f/6e, 20f/12e), tres semillas y alpha .25/.75. HitRate@10, cobertura de catálogo, RMSE para modelos de ratings, coste de ajuste y parámetros. Las cifras y semillas están en `reports/`. Evaluación offline, una interacción por usuario, no prueba satisfacción.

## Cold start, cobertura y diversidad

El test excluye usuarios/ítems nuevos no conocidos al modelo. Los perfiles corto (<=34 ratings de entrenamiento, n=155) y amplio (>=167, n=139) se miden aparte. La ruta fría se comprueba con usuario sin historial y usuario nuevo con tres películas de historial; no se reporta HitRate para el caso sin historia. Cobertura mide catálogo único recomendado, no diversidad intralista ni exposición justa. TF-IDF de géneros puede sobre-especializar y empatar con etiquetas compartidas.

## Riesgos y mitigación

Sesgo de selección y popularidad; retroalimentación; listas repetitivas; factores latentes no explicables; ausencia de datos demográficos; dataset pequeño y cambiante. Mitigación de esta entrega: separación temporal, fallback probado, métricas por perfil, cobertura, tres semillas, tres configuraciones, Pareto y no publicar ratings. Se requiere evaluación online/usuario y análisis de diversidad antes de desplegar.
