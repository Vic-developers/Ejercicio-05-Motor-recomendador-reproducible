# Declaración de uso de IA

## Herramienta

OpenAI Codex (GPT-6), como apoyo a programación, revisión del diseño de evaluación y redacción. La responsabilidad académica por el código, los resultados, las referencias y la defensa corresponde al estudiante.

## Prompt relevante

Resumen fiel de la instrucción: «Completa el proyecto base de LAB08/LAB09 según el mandato del Ejercicio 05; integra evaluación reproducible de popularidad, contenido, factorización e híbrido con corte temporal, tres semillas, perfiles fríos, comparación Pareto, pruebas, documentación y PDF; publica en el repositorio indicado y no redistribuyas MovieLens».

## Verificaciones y correcciones

Se descargó MovieLens directamente desde GroupLens, se verificó la auditoría local y el SHA-256, se ejecutó LAB08, se midieron los modelos en tres semillas y 587 eventos elegibles, se corrieron las siete pruebas y se generaron el notebook ejecutado y el PDF. La ejecución llevó a corregir el desempate temporal, explicitar la cohorte evaluable, resolver el índice del baseline de popularidad y pasar un vector compatible a la similitud coseno. Los resultados reportados provienen de los archivos locales generados por esos comandos; no se inventaron citas ni métricas. El MovieLens original no forma parte del repositorio.
