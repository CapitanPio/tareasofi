# Evaluación Regular N°2: modelos predictivos

Informe educativo sobre rendimiento académico y colocación laboral de estudiantes, elaborado con regresión lineal múltiple y regresión logística.

## Archivos

- [reporte_rendimiento_estudiantil.ipynb](reporte_rendimiento_estudiantil.ipynb): notebook con cálculos, tablas, gráficos, interpretaciones y conclusiones. Conserva las salidas del notebook ejecutado que se aportó para la revisión.
- [codigo_por_puntos.py](codigo_por_puntos.py): los mismos cálculos organizados por secciones, con comentarios breves.

## Ejecutar en Google Colab

1. [Abrir el notebook en Colab](https://colab.research.google.com/github/CapitanPio/tareasofi/blob/main/reporte_rendimiento_estudiantil.ipynb).
2. Ejecutar las celdas en orden. La primera instala las librerías necesarias.
3. Dejar `ruta_csv_local` vacía para descargar el dataset con KaggleHub. Para usar un CSV subido manualmente, escribir su ruta, por ejemplo `/content/student_dataset_10000_rows.csv`.
4. Completar los nombres de integrantes y la fecha de entrega de la portada antes de descargar el notebook.

## Ejecutar el código Python

```bash
python -m pip install kagglehub pandas numpy matplotlib seaborn scikit-learn ipython
python codigo_por_puntos.py
```

Para usar una copia local del dataset, completar `ruta_csv_local` en el archivo antes de ejecutarlo. La instalación requiere acceso a los paquetes; la descarga del dataset requiere acceso a Kaggle.

## Datos y metodología

Se utiliza [Student Performance Prediction Dataset](https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset), con 10.000 registros y ocho variables.

El CSV llama `attendance` y `previous_score` a las variables que la pauta denomina `Attendance Percentage` y `Previous Scores`. El código normaliza los nombres y resuelve esas dos diferencias sin modificar los valores.

Se divide la muestra en 80 % de entrenamiento y 20 % de prueba, con semilla 42 y estratificación por colocación laboral. La imputación y la estandarización se ajustan exclusivamente con entrenamiento. Se mantienen seis predictores y se compara cada modelo con una referencia sencilla.

## Resultados de la ejecución revisada

| Modelo | Resultados en prueba |
|---|---|
| Regresión lineal | MAE: 6,917 puntos; RMSE: 8,791 puntos; R²: 0,653. |
| Regresión logística | Accuracy: 90,65 %; precision: 93,09 %; recall: 95,93 %. |

Los porcentajes de clasificación se calcularon con los conteos de la matriz de confusión. La clase positiva es colocado.

Las conclusiones discuten el efecto techo en 100 puntos, las diferencias descriptivas entre grupos de estudio y sueño, y los 119 estudiantes no colocados clasificados como colocados. Los resultados representan asociaciones y no demuestran causalidad ni garantizan desempeño en otra institución.

Las interpretaciones corresponden al dataset y a la división evaluados. Si se cambian los datos, modelos o parámetros, se deben volver a ejecutar las celdas y revisar los textos numéricos.

La pauta solicita un notebook con salidas y una presentación en PDF de 10 a 15 minutos. El notebook incluye un guion de presentación; este repositorio contiene el informe y su código.
