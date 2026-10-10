"""Código por secciones del reporte de rendimiento estudiantil.

Las interpretaciones y salidas se encuentran en el notebook.
Para usar un CSV local, completar ruta_csv_local en la sección 2.
"""

# Instalar antes de ejecutar:
# python -m pip install kagglehub pandas numpy matplotlib seaborn scikit-learn ipython

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
from sklearn.impute import SimpleImputer
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

ruta_csv_local = ""

if ruta_csv_local == "":
    carpeta_dataset = kagglehub.dataset_download(
        "shambhurajejagadale/student-performance-prediction-dataset"
    )
    ruta_dataset = Path(carpeta_dataset) / "student_dataset_10000_rows.csv"
else:
    ruta_dataset = Path(ruta_csv_local)

datos = pd.read_csv(ruta_dataset)
datos.columns = (
    datos.columns.str.strip()
    .str.replace(r"[_\s]+", " ", regex=True)
    .str.title()
)

# La pauta usa dos nombres diferentes a los del CSV.
datos = datos.rename(columns={
    "Attendance": "Attendance Percentage",
    "Previous Score": "Previous Scores",
})

if datos.columns.duplicated().any():
    raise ValueError(
        "Hay nombres de columnas repetidos; revisar el archivo."
    )

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

columnas_faltantes = []
for nombre_columna in columnas_requeridas:
    if nombre_columna not in datos.columns:
        columnas_faltantes.append(nombre_columna)

if len(columnas_faltantes) > 0:
    raise ValueError(
        "Revisar los nombres reales del CSV. Faltan: "
        + str(columnas_faltantes)
        + ". Columnas disponibles: "
        + str(datos.columns.tolist())
    )

print("Filas:", datos.shape[0])
print("Columnas:", datos.shape[1])
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

eliminar_duplicados_exactos = True
filas_iniciales = len(datos)

if eliminar_duplicados_exactos:
    datos_sin_duplicados = datos.drop_duplicates().copy()
else:
    datos_sin_duplicados = datos.copy()

filas_duplicadas_eliminadas = filas_iniciales - len(datos_sin_duplicados)
datos_limpios = datos_sin_duplicados[columnas_requeridas].copy()

columnas_numericas = variables_predictoras + [variable_objetivo_lineal]

for nombre_columna in columnas_numericas:
    datos_limpios[nombre_columna] = pd.to_numeric(
        datos_limpios[nombre_columna],
        errors="coerce",
    )
    datos_limpios[nombre_columna] = datos_limpios[nombre_columna].replace(
        [np.inf, -np.inf],
        np.nan,
    )

estado_laboral_texto = (
    datos_limpios[variable_objetivo_logistica]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.replace(r"[\s_-]+", "", regex=True)
)
estado_laboral_texto = estado_laboral_texto.replace("", pd.NA)

codificacion_estado_laboral = {
    "placed": 1,
    "notplaced": 0,
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "1.0": 1,
    "0.0": 0,
}

categorias_desconocidas = estado_laboral_texto[
    estado_laboral_texto.notna()
    & ~estado_laboral_texto.isin(codificacion_estado_laboral)
].unique()

if len(categorias_desconocidas) > 0:
    raise ValueError(
        "Revisar la codificación de Placement Status: "
        + str(categorias_desconocidas)
    )

datos_limpios[variable_objetivo_logistica] = estado_laboral_texto.map(
    codificacion_estado_laboral
)

limites_fisicos = {
    "Study Hours": (0, 24),
    "Attendance Percentage": (0, 100),
    "Sleep Hours": (0, 24),
    "Internet Usage": (0, 24),
}

valores_invalidos = []

for nombre_variable, limites in limites_fisicos.items():
    limite_inferior = limites[0]
    limite_superior = limites[1]
    mascara_invalida = (
        (datos_limpios[nombre_variable] < limite_inferior)
        | (datos_limpios[nombre_variable] > limite_superior)
    )
    valores_invalidos.append({
        "Variable": nombre_variable,
        "Valores fuera del rango físico": int(mascara_invalida.sum()),
    })
    datos_limpios.loc[mascara_invalida, nombre_variable] = np.nan

mascara_tareas_invalidas = datos_limpios["Assignments Completed"] < 0
valores_invalidos.append({
    "Variable": "Assignments Completed",
    "Valores fuera del rango físico": int(mascara_tareas_invalidas.sum()),
})
datos_limpios.loc[mascara_tareas_invalidas, "Assignments Completed"] = np.nan

filas_antes_de_revisar_objetivos = len(datos_limpios)
datos_limpios = datos_limpios.dropna(
    subset=[variable_objetivo_lineal, variable_objetivo_logistica]
).copy()
datos_limpios[variable_objetivo_logistica] = (
    datos_limpios[variable_objetivo_logistica].astype(int)
)

filas_sin_objetivo_eliminadas = (
    filas_antes_de_revisar_objetivos - len(datos_limpios)
)

print("Duplicados eliminados:", filas_duplicadas_eliminadas)
print("Filas sin resultados eliminadas:", filas_sin_objetivo_eliminadas)
print("Filas disponibles:", len(datos_limpios))
display(pd.DataFrame(valores_invalidos))
display(datos_limpios[columnas_requeridas].isna().sum().to_frame("Faltantes"))

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

conteo_clases = datos_limpios[variable_objetivo_logistica].value_counts()

if len(conteo_clases) != 2 or conteo_clases.min() < 2:
    raise ValueError(
        "Se necesitan las dos clases laborales y suficientes observaciones "
        "para realizar una división estratificada."
    )

datos_entrenamiento, datos_prueba = train_test_split(
    datos_limpios,
    test_size=proporcion_prueba,
    random_state=semilla_aleatoria,
    stratify=datos_limpios[variable_objetivo_logistica],
)

print("Estudiantes en entrenamiento:", len(datos_entrenamiento))
print("Estudiantes en prueba:", len(datos_prueba))

proporciones_clases = pd.DataFrame({
    "Entrenamiento": datos_entrenamiento[variable_objetivo_logistica]
        .value_counts(normalize=True),
    "Prueba": datos_prueba[variable_objetivo_logistica]
        .value_counts(normalize=True),
})
display(proporciones_clases)

variables_utilizadas = variables_predictoras.copy()
variables_excluidas = []

for nombre_variable in variables_predictoras:
    cantidad_valores_distintos = datos_entrenamiento[nombre_variable].nunique()
    if cantidad_valores_distintos <= 1:
        variables_utilizadas.remove(nombre_variable)
        variables_excluidas.append(nombre_variable)

if len(variables_utilizadas) == 0:
    raise ValueError("No quedan predictores con variación para entrenar.")

print("Predictores utilizados:", variables_utilizadas)
print("Predictores sin variación suficiente:", variables_excluidas)

predictores_entrenamiento = datos_entrenamiento[variables_utilizadas]
predictores_prueba = datos_prueba[variables_utilizadas]

puntaje_entrenamiento = datos_entrenamiento[variable_objetivo_lineal]
puntaje_prueba = datos_prueba[variable_objetivo_lineal]

colocacion_entrenamiento = datos_entrenamiento[variable_objetivo_logistica]
colocacion_prueba = datos_prueba[variable_objetivo_logistica]

imputador = SimpleImputer(strategy="median")
predictores_entrenamiento_completos = imputador.fit_transform(
    predictores_entrenamiento
)
predictores_prueba_completos = imputador.transform(predictores_prueba)

tabla_medianas = pd.DataFrame({
    "Variable": variables_utilizadas,
    "Mediana de entrenamiento": imputador.statistics_,
})
display(tabla_medianas)

estandarizador = StandardScaler()
predictores_entrenamiento_estandarizados = estandarizador.fit_transform(
    predictores_entrenamiento_completos
)
predictores_prueba_estandarizados = estandarizador.transform(
    predictores_prueba_completos
)

# 7. Hipótesis adicional y relaciones básicas

datos_hipotesis = datos_entrenamiento.dropna(
    subset=["Study Hours", "Sleep Hours"]
).copy()

if len(datos_hipotesis) == 0:
    print(
        "No hay datos completos de estudio y sueño para esta comparación."
    )
else:
    mediana_horas_estudio = datos_entrenamiento["Study Hours"].median()

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

    comparacion_hipotesis = (
        datos_hipotesis
        .groupby(["Grupo de sueño", "Grupo de estudio"])["Exam Score"]
        .agg(["count", "mean", "std"])
        .rename(columns={
            "count": "Cantidad de estudiantes",
            "mean": "Puntaje promedio",
            "std": "Desviación estándar",
        })
    )

    print("Mediana de horas de estudio:", mediana_horas_estudio)
    print("Estudiantes utilizados en la comparación:", len(datos_hipotesis))
    display(comparacion_hipotesis)

correlaciones_entrenamiento = datos_entrenamiento[
    columnas_numericas
].corr()
display(correlaciones_entrenamiento)

figura, ejes = plt.subplots(1, 3, figsize=(15, 4))
variables_relacion_puntaje = [
    "Study Hours",
    "Attendance Percentage",
    "Previous Scores",
]

for posicion, nombre_variable in enumerate(variables_relacion_puntaje):
    sns.regplot(
        data=datos_entrenamiento,
        x=nombre_variable,
        y="Exam Score",
        ci=None,
        scatter_kws={"alpha": 0.15, "s": 8},
        line_kws={"color": "red"},
        ax=ejes[posicion],
    )

plt.tight_layout()
plt.show()

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

modelo_logistico = LogisticRegression(
    max_iter=2000,
    random_state=semilla_aleatoria,
)
modelo_logistico.fit(
    predictores_entrenamiento_estandarizados,
    colocacion_entrenamiento,
)

probabilidades_colocacion = modelo_logistico.predict_proba(
    predictores_prueba_estandarizados
)[:, 1]

umbral_clasificacion = 0.50
colocaciones_predichas = (
    probabilidades_colocacion >= umbral_clasificacion
).astype(int)

coeficientes_logisticos_originales = (
    modelo_logistico.coef_[0] / estandarizador.scale_
)

coeficientes_logisticos = pd.DataFrame({
    "Variable": variables_utilizadas,
    "Coeficiente en unidades originales": coeficientes_logisticos_originales,
    "Odds ratio por una unidad": np.exp(coeficientes_logisticos_originales),
})
display(coeficientes_logisticos)

exactitud = accuracy_score(colocacion_prueba, colocaciones_predichas)
precision = precision_score(
    colocacion_prueba,
    colocaciones_predichas,
    zero_division=0,
)
sensibilidad = recall_score(
    colocacion_prueba,
    colocaciones_predichas,
    zero_division=0,
)

modelo_referencia_logistico = DummyClassifier(strategy="most_frequent")
modelo_referencia_logistico.fit(
    predictores_entrenamiento_estandarizados,
    colocacion_entrenamiento,
)
colocaciones_referencia = modelo_referencia_logistico.predict(
    predictores_prueba_estandarizados
)

metricas_logisticas = pd.DataFrame({
    "Modelo": ["Regresión logística", "Clase más frecuente"],
    "Accuracy": [
        exactitud,
        accuracy_score(colocacion_prueba, colocaciones_referencia),
    ],
    "Precision": [
        precision,
        precision_score(
            colocacion_prueba,
            colocaciones_referencia,
            zero_division=0,
        ),
    ],
    "Recall": [
        sensibilidad,
        recall_score(
            colocacion_prueba,
            colocaciones_referencia,
            zero_division=0,
        ),
    ],
})

display(metricas_logisticas)

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

# 10. Revisión de supuestos y variables relevantes

puntajes_ajustados_entrenamiento = modelo_lineal.predict(
    predictores_entrenamiento_completos
)
residuos_entrenamiento = (
    puntaje_entrenamiento.to_numpy() - puntajes_ajustados_entrenamiento
)

plt.figure(figsize=(7, 4))
plt.scatter(
    puntajes_ajustados_entrenamiento,
    residuos_entrenamiento,
    alpha=0.20,
    s=10,
)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Puntaje estimado en entrenamiento")
plt.ylabel("Residuo en puntos")
plt.title("Residuos de la regresión lineal")
plt.tight_layout()
plt.show()

efectos_lineales_comparables = (
    modelo_lineal.coef_ * estandarizador.scale_
)

comparacion_variables = pd.DataFrame({
    "Variable": variables_utilizadas,
    "Cambio de puntaje por una desviación estándar": efectos_lineales_comparables,
    "Coeficiente logístico por una desviación estándar": modelo_logistico.coef_[0],
})

comparacion_variables["Magnitud del efecto lineal"] = np.abs(
    comparacion_variables["Cambio de puntaje por una desviación estándar"]
)
comparacion_variables["Magnitud del efecto logístico"] = np.abs(
    comparacion_variables["Coeficiente logístico por una desviación estándar"]
)

print("Orden descriptivo de efectos dentro de la regresión lineal:")
display(
    comparacion_variables.sort_values(
        "Magnitud del efecto lineal",
        ascending=False,
    )
)

print("Orden descriptivo de efectos dentro de la regresión logística:")
display(
    comparacion_variables.sort_values(
        "Magnitud del efecto logístico",
        ascending=False,
    )
)

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

