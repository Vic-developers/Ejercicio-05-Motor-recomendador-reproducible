from pathlib import Path
import json
import nbformat
import contextlib, io
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
ROOT=Path(__file__).resolve().parents[1]; rep=ROOT/'reports'; rep.mkdir(exist_ok=True)
main=pd.read_csv(rep/'main_results.csv'); pareto=pd.read_csv(rep/'pareto_comparison.csv'); seeds=pd.read_csv(rep/'seed_metrics.csv'); audit=json.loads((rep/'data_audit.json').read_text(encoding='utf-8'))
# Create and execute a compact evidence notebook based on the saved, reproducible experiment outputs.
nb=nbformat.v4.new_notebook(); cells=[
 nbformat.v4.new_markdown_cell('# Ejercicio 05 · Motor recomendador reproducible\n\nNotebook de evidencia del experimento LAB08/LAB09. Los datos MovieLens permanecen locales y no se redistribuyen. Resultados calculados con las semillas 17, 42 y 73; corte temporal por usuario.'),
 nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd, json\nroot=Path('..') if Path('../reports/main_results.csv').exists() else Path('.')\nreports=root/'reports'\nresults=pd.read_csv(reports/'main_results.csv')\npareto=pd.read_csv(reports/'pareto_comparison.csv')\nseed_runs=pd.read_csv(reports/'seed_metrics.csv')\nauditoría=json.loads((reports/'data_audit.json').read_text(encoding='utf-8'))\nprint(f\"MovieLens Latest Small: {auditoría['ratings']:,} ratings, {auditoría['movies']:,} películas, {auditoría['users']} usuarios\")\nprint(f\"Eventos temporales elegibles: {auditoría['eligible_test_events']}\")\nprint(f\"SHA-256 ZIP: {auditoría['sha256']}\")"),
 nbformat.v4.new_markdown_cell('## Tabla principal\n\nHitRate@10 usa un ítem retenido por usuario. Cobertura es la proporción de películas del catálogo que aparece en alguna lista. RMSE solo aplica a modelos de rating. El baseline de contenido expresa coincidencia de géneros, no satisfacción.'),
 nbformat.v4.new_code_cell("print(results.round(4).to_string(index=False))"),
 nbformat.v4.new_markdown_cell('## Comparación Pareto del híbrido\n\nSe maximiza HitRate@10 y cobertura, y se minimiza tiempo de entrenamiento y número de parámetros. La decisión debe considerar el objetivo: alpha alto prioriza aciertos; alpha bajo amplía cobertura.'),
 nbformat.v4.new_code_cell("print(pareto.round(4).to_string(index=False))"),
 nbformat.v4.new_markdown_cell('## Interpretación y límites\n\nEn esta corrida alpha=0.75 igualó el HitRate@10 colaborativo con cobertura ligeramente mayor; alpha=0.25 obtuvo mayor cobertura, con una pequeña reducción de aciertos. Ambos pesos son Pareto eficientes. Se elige alpha=0.75 si el objetivo primario es recuperar el ítem retenido, y se conserva alpha=0.25 como alternativa de mayor exposición de catálogo. Solo se evalúan 587 usuarios/ítems conocidos por entrenamiento: cold start queda fuera de la métrica. Los resultados no miden satisfacción real ni significancia estadística. MovieLens contiene ratings autoseleccionados, carece de demografía y puede cambiar entre descargas.')]
nb.cells=cells; nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13'}}
ROOT.joinpath('notebooks').mkdir(exist_ok=True)
ns={}
for count,cell in enumerate(nb.cells,1):
 if cell.cell_type=='code':
  out=io.StringIO()
  with contextlib.redirect_stdout(out): exec(cell.source,ns)
  cell.execution_count=count; cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=out.getvalue())]
nbformat.write(nb,ROOT/'notebooks/ejercicio_05_motor_recomendador.ipynb')
# Build the concise PDF hand-in.
styles=getSampleStyleSheet(); styles['BodyText'].fontSize=9; styles['BodyText'].leading=11; styles['BodyText'].spaceAfter=4; styles['Heading2'].fontSize=12; styles['Heading2'].leading=14; styles['Heading2'].spaceBefore=7; styles['Heading2'].spaceAfter=4; styles.add(ParagraphStyle(name='TitleCenter',parent=styles['Title'],fontSize=17,leading=20,alignment=TA_CENTER,textColor=colors.HexColor('#17365D'))); styles.add(ParagraphStyle(name='Small',parent=styles['BodyText'],fontSize=7.5,leading=9)); story=[]
story += [Paragraph('Ejercicio 05 · Motor recomendador reproducible',styles['TitleCenter']),Paragraph('INF-8239 · Unidad 03 · Edwin Ramón José Nolasco',styles['Normal']),Spacer(1,5)]
story += [Paragraph('Repositorio de entrega',styles['Heading2']),Paragraph('<link href="https://github.com/Vic-developers/Ejercicio-05-Motor-recomendador-reproducible" color="blue">github.com/Vic-developers/Ejercicio-05-Motor-recomendador-reproducible</link>',styles['BodyText']),Spacer(1,4)]
story += [Paragraph('Datos y protocolo',styles['Heading2']),Paragraph(f"MovieLens Latest Small: {audit['ratings']:,} ratings, {audit['movies']:,} películas y {audit['users']} usuarios; densidad {audit['density']:.2%}. SHA-256 del ZIP: {audit['sha256']}. Se retuvo una última valoración por usuario; se evaluaron {audit['eligible_test_events']} eventos cuyo usuario e ítem estaban representados en entrenamiento. Semillas 17, 42 y 73; 20 factores y 12 épocas.",styles['BodyText']),Spacer(1,4)]
story += [Paragraph('Resultados principales',styles['Heading2'])]
cols=['Método','RMSE','HitRate@10','Cobertura','Ajuste (s)']
rows=[cols]
labels={'popularity':'Popularidad','content':'Contenido','collaborative':'Colaborativo','hybrid_alpha_0.25':'Híbrido alpha=.25','hybrid_alpha_0.75':'Híbrido alpha=.75'}
for _,r in main.iterrows(): rows.append([labels.get(r.method,r.method),'—' if pd.isna(r.rmse) else f"{r.rmse:.3f}",f"{r.hit_rate_at_10:.3f}",f"{r.catalog_coverage:.1%}",f"{r.fit_seconds:.1f}"])
t=Table(rows,colWidths=[1.55*inch,.75*inch,1.05*inch,1.0*inch,.9*inch],repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17365D')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#9EADBD')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EDF2F7')]),('ALIGN',(1,1),(-1,-1),'RIGHT'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); story += [t,Spacer(1,4)]
story += [Paragraph('Lectura y decisión',styles['Heading2']),Paragraph('El baseline de contenido cubre más catálogo (14.9%) que el híbrido, pero no recuperó el único ítem retenido en esta evaluación. Popularidad logra 0.2% de HitRate@10 y solo 0.13% de cobertura porque concentra las listas. El colaborativo obtiene 0.031 de HitRate@10. alpha=.75 mantiene esa tasa media y mejora ligeramente la cobertura colaborativa (7.6% vs. 7.4%); alpha=.25 amplía cobertura a 10.4% y reduce HitRate a 0.029. Ambos híbridos son Pareto eficientes. Se selecciona alpha=.75 si prima el acierto del ítem retenido; alpha=.25 es la opción de más cobertura. La diferencia es pequeña y no demuestra significancia con tres semillas.',styles['BodyText']),Spacer(1,4)]
story += [Paragraph('Límites y cold start',styles['Heading2']),Paragraph('RMSE aproxima ratings retenidos y no garantiza un ranking útil. El corte excluye 23 eventos no evaluables porque su ítem no aparece en entrenamiento; no se afirma desempeño cold start. El fallback frío es popularidad global, funcional pero no personalizado. Géneros amplios pueden causar empates y sobre-especialización. El dataset está autoseleccionado, no tiene demografía y su versión “latest” cambia. No se debe interpretar el resultado como satisfacción, causalidad ni preferencia de grupos.',styles['BodyText']),Spacer(1,4)]
story += [Paragraph('Reproducibilidad, pruebas y uso de IA',styles['Heading2']),Paragraph('El notebook ejecutado, `src/`, `tests/`, `reports/`, README, requirements.txt y scripts acompañan este PDF. MovieLens no se incluye en Git. Se usó OpenAI Codex (GPT-6) para depuración, protocolo y redacción, a partir del mandato de la tarea; sus sugerencias se verificaron ejecutando auditoría de datos, LAB08, evaluación en tres semillas, pruebas automatizadas y generación del notebook. Correcciones verificadas: desempate temporal por movieId, coherencia de cohorte, coseno con vectores compatibles y exclusión de los datos del control de versiones. El estudiante responde por explicar y defender cada resultado.',styles['Small']),Spacer(1,3),Paragraph('Fuentes: GroupLens Research, MovieLens Latest Small README: <link href="https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html" color="blue">files.grouplens.org/.../ml-latest-small-README.html</link> Harper & Konstan (2015), The MovieLens Datasets: History and Context, DOI 10.1145/2827872.',styles['Small'])]
doc=SimpleDocTemplate(str(rep/'ejercicio_05_informe.pdf'),pagesize=letter,rightMargin=0.55*inch,leftMargin=0.55*inch,topMargin=0.5*inch,bottomMargin=0.5*inch,title='Ejercicio 05 - Motor recomendador reproducible',author='Edwin Ramón José Nolasco'); doc.build(story)
print('Notebook ejecutado y PDF generados.')
