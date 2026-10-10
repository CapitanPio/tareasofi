# Código del notebook de rendimiento estudiantil.
# Instalar antes: kagglehub pandas numpy matplotlib seaborn scikit-learn ipython

# 2. Librerías y carga del dataset

from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:.3f}".format)

semilla_aleatoria = 42
proporcion_prueba = 0.20

carpeta_dataset = kagglehub.dataset_download(
    "shambhurajejagadale/student-performance-prediction-dataset"
)
ruta_dataset = Path(carpeta_dataset) / "student_dataset_10000_rows.csv"

datos = pd.read_csv(ruta_dataset)
datos.columns = (
    datos.columns.str.strip()
    .str.replace(r"[_\s]+", " ", regex=True)
    .str.title()
)

# La pauta usa dos nombres diferentes a los del CSV, los vamos a cambiar para que sean coherentes con la pauta.
datos = datos.rename(columns={
    "Attendance": "Attendance Percentage",
    "Previous Score": "Previous Scores",
})

variables_predictoras = [
    "Study Hours",
    "Attendance Percentage",
    "Sleep Hours",
    "Internet Usage",
    "Assignments Completed",
    "Previous Scores",
]

variable_objetivo_lineal = "Exam Score"
variable_objetivo_logistica = "Placement Status"

columnas_requeridas = variables_predictoras + [
    variable_objetivo_lineal,
    variable_objetivo_logistica,
]


print("Filas:", datos.shape[0])
print("Columnas:", datos.shape[1])

# Una muestra visual de como son los dátos
display(datos.head())

# 3. Exploración básica antes de limpiar

resumen_inicial = pd.DataFrame({
    "Tipo de dato": datos.dtypes.astype(str),
    "Valores faltantes": datos.isna().sum(),
    "Porcentaje faltante": datos.isna().mean() * 100,
})

display(resumen_inicial)
display(datos.describe())
print("Filas completamente duplicadas:", datos.duplicated().sum())
display(datos[variable_objetivo_logistica].value_counts(dropna=False))

variables_graficos = ["Study Hours", "Sleep Hours", "Exam Score"]
figura, ejes = plt.subplots(1, 3, figsize=(15, 4))

for posicion, nombre_variable in enumerate(variables_graficos):
    valores_numericos = pd.to_numeric(
        datos[nombre_variable],
        errors="coerce",
    )
    sns.histplot(
        valores_numericos.dropna(),
        bins=20,
        ax=ejes[posicion],
    )
    ejes[posicion].set_title(nombre_variable)
    ejes[posicion].set_ylabel("Cantidad de estudiantes")

plt.tight_layout()
plt.show()

# 4. Limpieza y selección de variables

datos_limpios = datos[columnas_requeridas].copy()

columnas_numericas = variables_predictoras + [
    variable_objetivo_lineal
]

codificacion_estado_laboral = {
    "Not Placed": 0,
    "Placed": 1,
}

datos_limpios[variable_objetivo_logistica] = (
    datos_limpios[variable_objetivo_logistica]
    .map(codificacion_estado_laboral)
)

print("Registros conservados:", len(datos_limpios))
display(datos_limpios[variable_objetivo_logistica].value_counts())

# 5. Valores atípicos

resumen_atipicos = []

for nombre_variable in columnas_numericas:
    valores = datos_limpios[nombre_variable].dropna()
    primer_cuartil = valores.quantile(0.25)
    tercer_cuartil = valores.quantile(0.75)
    rango_intercuartilico = tercer_cuartil - primer_cuartil

    limite_inferior = primer_cuartil - 1.5 * rango_intercuartilico
    limite_superior = tercer_cuartil + 1.5 * rango_intercuartilico

    cantidad_atipicos = (
        (valores < limite_inferior)
        | (valores > limite_superior)
    ).sum()

    resumen_atipicos.append({
        "Variable": nombre_variable,
        "Límite inferior IQR": limite_inferior,
        "Límite superior IQR": limite_superior,
        "Cantidad señalada": int(cantidad_atipicos),
    })

display(pd.DataFrame(resumen_atipicos))

figura, ejes = plt.subplots(1, 3, figsize=(15, 4))

for posicion, nombre_variable in enumerate(variables_graficos):
    sns.boxplot(
        y=datos_limpios[nombre_variable],
        ax=ejes[posicion],
    )
    ejes[posicion].set_title(nombre_variable)

plt.tight_layout()
plt.show()

# 6. División de entrenamiento y prueba

# Dividir los datos en entrenamiento y prueba.
datos_entrenamiento, datos_prueba = train_test_split(
    datos_limpios,
    test_size=proporcion_prueba,
    random_state=semilla_aleatoria,
    stratify=datos_limpios[variable_objetivo_logistica],
)

variables_utilizadas = variables_predictoras.copy()

# Seleccionar las variables predictoras.
predictores_entrenamiento_completos = (
    datos_entrenamiento[variables_utilizadas]
)

predictores_prueba_completos = (
    datos_prueba[variables_utilizadas]
)

# Separar los resultados que queremos predecir.
puntaje_entrenamiento = (
    datos_entrenamiento[variable_objetivo_lineal]
)

puntaje_prueba = (
    datos_prueba[variable_objetivo_lineal]
)

colocacion_entrenamiento = (
    datos_entrenamiento[variable_objetivo_logistica]
)

colocacion_prueba = (
    datos_prueba[variable_objetivo_logistica]
)

# Estandarizar los predictores para la regresión logística.
estandarizador = StandardScaler()

predictores_entrenamiento_estandarizados = (
    estandarizador.fit_transform(
        predictores_entrenamiento_completos
    )
)

predictores_prueba_estandarizados = (
    estandarizador.transform(
        predictores_prueba_completos
    )
)

print("Estudiantes en entrenamiento:", len(datos_entrenamiento))
print("Estudiantes en prueba:", len(datos_prueba))

# 7. Hipótesis adicional y relaciones básicas

# Organizar los tres gráficos en una sola fila.
figura, ejes = plt.subplots(
    1,
    3,
    figsize=(18, 5),
    sharey=True,
)

# Primer gráfico: puntajes agrupados por horas de estudio.
sns.boxplot(
    data=datos_entrenamiento,
    x="Study Hours",
    y="Exam Score",
    ax=ejes[0],
)

ejes[0].set_title("Horas de estudio y puntaje")
ejes[0].set_xlabel("Horas de estudio diarias")
ejes[0].set_ylabel("Puntaje del examen")

# Segundo gráfico: asistencia y puntaje.
sns.regplot(
    data=datos_entrenamiento,
    x="Attendance Percentage",
    y="Exam Score",
    ci=None,
    scatter_kws={"alpha": 0.15},
    line_kws={"color": "red"},
    ax=ejes[1],
)

ejes[1].set_title("Asistencia y puntaje")
ejes[1].set_xlabel("Asistencia (%)")
ejes[1].set_ylabel("")

# Tercer gráfico: puntaje previo y puntaje del examen.
sns.regplot(
    data=datos_entrenamiento,
    x="Previous Scores",
    y="Exam Score",
    ci=None,
    scatter_kws={"alpha": 0.15},
    line_kws={"color": "red"},
    ax=ejes[2],
)

ejes[2].set_title("Puntaje previo y examen")
ejes[2].set_xlabel("Puntaje previo")
ejes[2].set_ylabel("")

plt.tight_layout()
plt.show()

# Agrupar estudiantes según sus horas de estudio y sueño.
datos_hipotesis = datos_entrenamiento.copy()

mediana_horas_estudio = datos_hipotesis["Study Hours"].median()

datos_hipotesis["Grupo de estudio"] = np.where(
    datos_hipotesis["Study Hours"] >= mediana_horas_estudio,
    "Igual o superior a la mediana",
    "Inferior a la mediana",
)

datos_hipotesis["Grupo de sueño"] = np.where(
    datos_hipotesis["Sleep Hours"] < 6,
    "Menos de 6 horas",
    "6 horas o más",
)

# Comparar cantidad de estudiantes, promedio y dispersión.
comparacion_hipotesis = datos_hipotesis.groupby(
    ["Grupo de sueño", "Grupo de estudio"]
)["Exam Score"].agg(["count", "mean", "std"])

comparacion_hipotesis = comparacion_hipotesis.rename(columns={
    "count": "Cantidad de estudiantes",
    "mean": "Puntaje promedio",
    "std": "Desviación estándar",
})

print("Mediana de horas de estudio:", mediana_horas_estudio)
display(comparacion_hipotesis)

# Revisar las correlaciones entre las variables numéricas.
correlaciones_entrenamiento = (
    datos_entrenamiento[columnas_numericas].corr()
)

display(correlaciones_entrenamiento)

medianas_por_hora = datos_entrenamiento.groupby(
    "Study Hours"
)["Exam Score"].median()

display(medianas_por_hora)

# Organizar los promedios como una tabla dinámica.
promedios_por_grupo = datos_hipotesis.pivot_table(
    index="Grupo de sueño",
    columns="Grupo de estudio",
    values="Exam Score",
    aggfunc="mean",
)

# Calcular la diferencia entre estudio alto y bajo.
promedios_por_grupo["Diferencia de puntaje"] = (
    promedios_por_grupo["Igual o superior a la mediana"]
    - promedios_por_grupo["Inferior a la mediana"]
)

display(promedios_por_grupo.round(2))

# 8. Regresión lineal múltiple

modelo_lineal = LinearRegression()
modelo_lineal.fit(
    predictores_entrenamiento_completos,
    puntaje_entrenamiento,
)

puntajes_predichos = modelo_lineal.predict(predictores_prueba_completos)

coeficientes_lineales = pd.DataFrame({
    "Variable": variables_utilizadas,
    "Cambio de puntaje por una unidad": modelo_lineal.coef_,
})

print("Intercepto del modelo lineal:", modelo_lineal.intercept_)
display(coeficientes_lineales)

error_absoluto_medio = mean_absolute_error(
    puntaje_prueba,
    puntajes_predichos,
)
raiz_error_cuadratico_medio = np.sqrt(
    mean_squared_error(puntaje_prueba, puntajes_predichos)
)
coeficiente_determinacion = r2_score(
    puntaje_prueba,
    puntajes_predichos,
)

modelo_referencia_lineal = DummyRegressor(strategy="mean")
modelo_referencia_lineal.fit(
    predictores_entrenamiento_completos,
    puntaje_entrenamiento,
)
puntajes_referencia = modelo_referencia_lineal.predict(
    predictores_prueba_completos
)

metricas_lineales = pd.DataFrame({
    "Modelo": ["Regresión lineal", "Promedio de entrenamiento"],
    "MAE": [
        error_absoluto_medio,
        mean_absolute_error(puntaje_prueba, puntajes_referencia),
    ],
    "RMSE": [
        raiz_error_cuadratico_medio,
        np.sqrt(mean_squared_error(puntaje_prueba, puntajes_referencia)),
    ],
    "R cuadrado": [
        coeficiente_determinacion,
        r2_score(puntaje_prueba, puntajes_referencia),
    ],
})

display(metricas_lineales)

# 9. Regresión logística

# Entrenar el modelo de regresión logística.
modelo_logistico = LogisticRegression(
    max_iter=2000,
    random_state=semilla_aleatoria,
)

modelo_logistico.fit(
    predictores_entrenamiento_estandarizados,
    colocacion_entrenamiento,
)

# Estimar la probabilidad de colocación laboral.
probabilidades_colocacion = modelo_logistico.predict_proba(
    predictores_prueba_estandarizados
)[:, 1]

umbral_clasificacion = 0.50

colocaciones_predichas = (
    probabilidades_colocacion >= umbral_clasificacion
).astype(int)

# Expresar los coeficientes en las unidades originales.
coeficientes_logisticos_originales = (
    modelo_logistico.coef_[0] / estandarizador.scale_
)

coeficientes_logisticos = pd.DataFrame({
    "Variable": variables_utilizadas,
    "Coeficiente en unidades originales": coeficientes_logisticos_originales,
    "Odds ratio por una unidad": np.exp(
        coeficientes_logisticos_originales
    ),
})

display(coeficientes_logisticos)

# Evaluar las predicciones del modelo.
exactitud = accuracy_score(
    colocacion_prueba,
    colocaciones_predichas,
)

precision = precision_score(
    colocacion_prueba,
    colocaciones_predichas,
    pos_label=1,
)

sensibilidad = recall_score(
    colocacion_prueba,
    colocaciones_predichas,
    pos_label=1,
)

especificidad = recall_score(
    colocacion_prueba,
    colocaciones_predichas,
    pos_label=0,
)

# Crear una referencia que predice siempre la clase más frecuente.
modelo_referencia_logistico = DummyClassifier(
    strategy="most_frequent"
)

modelo_referencia_logistico.fit(
    predictores_entrenamiento_estandarizados,
    colocacion_entrenamiento,
)

colocaciones_referencia = modelo_referencia_logistico.predict(
    predictores_prueba_estandarizados
)

# Evaluar las predicciones de la referencia.
exactitud_referencia = accuracy_score(
    colocacion_prueba,
    colocaciones_referencia,
)

precision_referencia = precision_score(
    colocacion_prueba,
    colocaciones_referencia,
    pos_label=1,
)

sensibilidad_referencia = recall_score(
    colocacion_prueba,
    colocaciones_referencia,
    pos_label=1,
)

especificidad_referencia = recall_score(
    colocacion_prueba,
    colocaciones_referencia,
    pos_label=0,
)

# Comparar los resultados como porcentajes.
metricas_logisticas = pd.DataFrame({
    "Modelo": [
        "Regresión logística",
        "Clase más frecuente",
    ],
    "Accuracy (%)": [
        exactitud * 100,
        exactitud_referencia * 100,
    ],
    "Precision de colocados (%)": [
        precision * 100,
        precision_referencia * 100,
    ],
    "Sensibilidad: colocados (%)": [
        sensibilidad * 100,
        sensibilidad_referencia * 100,
    ],
    "Especificidad: no colocados (%)": [
        especificidad * 100,
        especificidad_referencia * 100,
    ],
})

display(metricas_logisticas.round(2))

# Mostrar los aciertos y errores de clasificación.
ConfusionMatrixDisplay.from_predictions(
    colocacion_prueba,
    colocaciones_predichas,
    labels=[0, 1],
    display_labels=["No colocado", "Colocado"],
    cmap="Blues",
    values_format="d",
)

plt.title("Matriz de confusión de la regresión logística")
plt.tight_layout()
plt.show()

# 10. Revisión de los modelos y variables relevantes

# Comparar los puntajes reales con los estimados.
plt.figure(figsize=(7, 5))

plt.scatter(
    puntaje_prueba,
    puntajes_predichos,
    alpha=0.20,
)

puntaje_minimo = min(puntaje_prueba.min(), puntajes_predichos.min())
puntaje_maximo = max(puntaje_prueba.max(), puntajes_predichos.max())

plt.plot(
    [puntaje_minimo, puntaje_maximo],
    [puntaje_minimo, puntaje_maximo],
    color="red",
)

plt.xlabel("Puntaje real")
plt.ylabel("Puntaje estimado")
plt.title("Comparación entre puntajes reales y estimados")
plt.tight_layout()
plt.show()

# Comparar los coeficientes considerando la escala de las variables.
efectos_lineales_comparables = (
    modelo_lineal.coef_ * estandarizador.scale_
)

comparacion_variables = pd.DataFrame({
    "Variable": variables_utilizadas,
    "Magnitud en regresión lineal": np.abs(efectos_lineales_comparables),
    "Magnitud en regresión logística": np.abs(modelo_logistico.coef_[0]),
})

orden_variables_lineales = comparacion_variables.sort_values(
    "Magnitud en regresión lineal",
    ascending=False,
)["Variable"].tolist()

orden_variables_logisticas = comparacion_variables.sort_values(
    "Magnitud en regresión logística",
    ascending=False,
)["Variable"].tolist()

orden_variables = pd.DataFrame({
    "Orden": range(1, len(variables_utilizadas) + 1),
    "Regresión lineal": orden_variables_lineales,
    "Regresión logística": orden_variables_logisticas,
})

display(orden_variables)

# 11. Predicciones para estudiantes no utilizados en entrenamiento

resultados_prueba = pd.DataFrame({
    "Fila original": datos_prueba.index,
    "Puntaje real": puntaje_prueba.to_numpy(),
    "Puntaje estimado": puntajes_predichos,
    "Colocación real": colocacion_prueba.to_numpy(),
    "Colocación estimada": colocaciones_predichas,
    "Probabilidad de colocación": probabilidades_colocacion,
})

resultados_prueba["Error absoluto del puntaje"] = np.abs(
    resultados_prueba["Puntaje real"] - resultados_prueba["Puntaje estimado"]
)

display(resultados_prueba.head(10))
