# Código por secciones del reporte de estudiantes.
# Instalar antes: python -m pip install kagglehub pandas numpy matplotlib seaborn scikit-learn ipython
# Este archivo no contiene resultados reales; ejecutar con el CSV original.
# # Rendimiento académico y colocación laboral de estudiantes
# 
# Integrantes: [Completar nombres]. USÉ IA PARA ESTO, NO LO HICE YO
# Asignatura: Modelos Predictivos en Negocios. USÉ IA PARA ESTO, NO LO HICE YO
# Fecha: [Completar fecha de entrega]. USÉ IA PARA ESTO, NO LO HICE YO
# 
# Este notebook es una propuesta educativa basada en la pauta Regular_02.pdf y debe ejecutarse con el CSV real antes de entregar el trabajo. USÉ IA PARA ESTO, NO LO HICE YO
# Las conclusiones numéricas deben escribirse con las salidas obtenidas al ejecutar el notebook; no se incluyen resultados precalculados. USÉ IA PARA ESTO, NO LO HICE YO
# 
# Ejecutar las celdas en orden en Google Colab y descargar el archivo .ipynb después de comprobar que conserva todas sus salidas. USÉ IA PARA ESTO, NO LO HICE YO

# ## 1. Contexto, objetivo e hipótesis
# 
# Una institución educativa puede utilizar información de sus estudiantes para estimar su rendimiento académico y orientar programas de apoyo y preparación laboral. USÉ IA PARA ESTO, NO LO HICE YO
# El objetivo es construir una regresión lineal múltiple para estimar Exam Score y una regresión logística para estimar la probabilidad de Placement Status. USÉ IA PARA ESTO, NO LO HICE YO
# 
# Hipótesis principal: más horas de estudio, mayor asistencia y mejores puntajes previos se relacionan con mejores resultados académicos y mayor probabilidad de colocación laboral. USÉ IA PARA ESTO, NO LO HICE YO
# Hipótesis exploratoria adicional: la relación entre estudiar más y obtener mejores puntajes podría ser distinta entre estudiantes que duermen menos de seis horas y quienes duermen seis horas o más. USÉ IA PARA ESTO, NO LO HICE YO
# La segunda hipótesis será una comparación descriptiva de grupos, no una prueba de causalidad ni una modificación de los modelos exigidos. USÉ IA PARA ESTO, NO LO HICE YO

# ## 2. Librerías y carga del dataset
# 
# Se utilizan pandas y numpy para manejar datos, matplotlib y seaborn para gráficos, scikit-learn para los modelos y kagglehub para descargar el archivo. USÉ IA PARA ESTO, NO LO HICE YO
# No se utiliza statsmodels. USÉ IA PARA ESTO, NO LO HICE YO

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

# Dejar ruta_csv_local vacía para descargar desde Kaggle; si la descarga falla, subir student_dataset_10000_rows.csv a Colab y escribir su ruta, por ejemplo /content/student_dataset_10000_rows.csv. USÉ IA PARA ESTO, NO LO HICE YO
# No se necesitan credenciales dentro del código para un dataset público y no se deben pegar claves privadas en el notebook. USÉ IA PARA ESTO, NO LO HICE YO

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

if datos.columns.duplicated().any():
    raise ValueError("Hay nombres de columnas repetidos; revisar el archivo.")

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

# ## 3. Exploración básica antes de limpiar
# 
# Se revisan tipos de datos, valores faltantes, duplicados y estadísticos básicos para decidir la limpieza. USÉ IA PARA ESTO, NO LO HICE YO
# Los gráficos se limitan a tres histogramas y diagramas de caja en la sección de valores atípicos, además de una tabla de correlaciones más adelante. USÉ IA PARA ESTO, NO LO HICE YO

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

# Interpretación por completar: indicar el número de filas, si hay valores faltantes o duplicados y qué se observa en las tres distribuciones. USÉ IA PARA ESTO, NO LO HICE YO
# No describir una distribución como normal solo por su apariencia ni escribir cifras que no estén en las salidas anteriores. USÉ IA PARA ESTO, NO LO HICE YO

# ## 4. Limpieza y selección de variables
# 
# Los duplicados exactos se consideran registros repetidos en esta propuesta; antes de eliminarlos, comprobar si existe un identificador que indique que pertenecen a estudiantes diferentes. USÉ IA PARA ESTO, NO LO HICE YO
# Si los duplicados corresponden a personas diferentes, cambiar eliminar_duplicados_exactos a False y justificar esa decisión. USÉ IA PARA ESTO, NO LO HICE YO
# Se eliminan filas sin alguno de los dos resultados que queremos predecir para poder utilizar la misma muestra en ambos modelos. USÉ IA PARA ESTO, NO LO HICE YO
# Los predictores faltantes se completarán con medianas calculadas exclusivamente en entrenamiento, conservando así los registros que sí tienen resultados observados. USÉ IA PARA ESTO, NO LO HICE YO

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

# Los rangos físicos anteriores se basan en horas diarias, asistencia porcentual y tareas no negativas, según las descripciones de la pauta. USÉ IA PARA ESTO, NO LO HICE YO
# No se impone una escala de 0 a 100 a los puntajes sin confirmar que esa sea su escala real. USÉ IA PARA ESTO, NO LO HICE YO
# Los valores que no se pueden convertir a número o que son imposibles se transforman en faltantes para tratarlos de forma explícita. USÉ IA PARA ESTO, NO LO HICE YO
# No se incluyen identificadores ni otras columnas no justificadas y se excluye Exam Score de los predictores de colocación para representar una predicción anterior al examen final. USÉ IA PARA ESTO, NO LO HICE YO

# ## 5. Valores atípicos
# 
# Se utiliza el rango intercuartílico o IQR para identificar observaciones poco frecuentes, sin eliminarlas automáticamente. USÉ IA PARA ESTO, NO LO HICE YO
# Un valor fuera de estos límites puede ser válido y no equivale necesariamente a un error. USÉ IA PARA ESTO, NO LO HICE YO
# Esta propuesta conserva los valores físicamente posibles para evitar descartar estudiantes con características distintas al promedio. USÉ IA PARA ESTO, NO LO HICE YO

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

# ## 6. División de entrenamiento y prueba
# 
# Se reserva el 80 % de los estudiantes para entrenamiento y el 20 % para evaluar resultados en casos que los modelos no utilizaron para aprender. USÉ IA PARA ESTO, NO LO HICE YO
# La división mantiene aproximadamente la proporción de colocados y no colocados en ambos grupos y se utiliza la misma semilla para reproducirla. USÉ IA PARA ESTO, NO LO HICE YO
# Las medianas, las variables constantes y la estandarización se determinan después de dividir, utilizando únicamente entrenamiento. USÉ IA PARA ESTO, NO LO HICE YO

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

# Estandarizar significa restar el promedio y dividir por la desviación estándar de entrenamiento; cambia las unidades, pero conserva la información y las correlaciones entre variables no constantes. USÉ IA PARA ESTO, NO LO HICE YO
# El centrado cambia el punto cero, por lo que no conserva cocientes entre valores como si fueran proporciones originales. USÉ IA PARA ESTO, NO LO HICE YO
# La regresión lineal utiliza las unidades originales para facilitar la explicación de sus coeficientes y la logística utiliza datos estandarizados para facilitar el ajuste numérico. USÉ IA PARA ESTO, NO LO HICE YO
# En ningún caso se usa la información de prueba para calcular medianas o escalas. USÉ IA PARA ESTO, NO LO HICE YO

# ## 7. Hipótesis adicional y relaciones básicas
# 
# Estas comparaciones utilizan únicamente entrenamiento y no cambian los modelos ni sirven para ajustar decisiones mirando el conjunto de prueba. USÉ IA PARA ESTO, NO LO HICE YO
# Se considera estudio alto un valor igual o superior a la mediana de horas de estudio de entrenamiento y sueño corto un valor inferior a seis horas diarias. USÉ IA PARA ESTO, NO LO HICE YO
# Las etiquetas alto y bajo son relativas a esta muestra, no recomendaciones sobre cuánto estudiar. USÉ IA PARA ESTO, NO LO HICE YO
# Se muestran cantidades, promedios y desviaciones estándar; grupos pequeños o muy variables permiten conclusiones menos firmes. USÉ IA PARA ESTO, NO LO HICE YO

datos_hipotesis = datos_entrenamiento.dropna(
    subset=["Study Hours", "Sleep Hours"]
).copy()

if len(datos_hipotesis) == 0:
    print(
        "No hay datos completos de estudio y sueño para esta comparación. "
        "USÉ IA PARA ESTO, NO LO HICE YO"
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

# Interpretación por completar: comparar la diferencia de puntajes entre estudio alto y bajo dentro de cada grupo de sueño, considerando sus tamaños y dispersión. USÉ IA PARA ESTO, NO LO HICE YO
# La comparación no controla otras diferencias entre estudiantes y no demuestra que dormir poco modifique causalmente el efecto de estudiar. USÉ IA PARA ESTO, NO LO HICE YO
# Las correlaciones y gráficos ayudan a describir relaciones y advertir predictores muy relacionados, pero una correlación baja no justifica por sí sola eliminar una variable. USÉ IA PARA ESTO, NO LO HICE YO

# ## 8. Regresión lineal múltiple
# 
# El modelo estima el puntaje del examen a partir de las seis características seleccionadas, salvo las que no presenten variación suficiente en entrenamiento. USÉ IA PARA ESTO, NO LO HICE YO
# Cada coeficiente indica el cambio estimado en el puntaje al aumentar una unidad del predictor, manteniendo constantes los demás. USÉ IA PARA ESTO, NO LO HICE YO
# En asistencia, una unidad corresponde a un punto porcentual y no a un aumento relativo de un uno por ciento. USÉ IA PARA ESTO, NO LO HICE YO

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

# MAE expresa el error absoluto promedio en puntos y ayuda a valorar si las predicciones son suficientemente precisas para orientar apoyos académicos. USÉ IA PARA ESTO, NO LO HICE YO
# RMSE también se expresa en puntos y da mayor peso a errores grandes, que podrían dejar sin apoyo a estudiantes que lo necesitan. USÉ IA PARA ESTO, NO LO HICE YO
# R cuadrado resume el desempeño frente a una predicción constante; puede ser negativo y no equivale al porcentaje de estudiantes correctamente predichos. USÉ IA PARA ESTO, NO LO HICE YO
# El modelo de referencia siempre predice el promedio de entrenamiento y permite comprobar si utilizar los predictores mejora esa estrategia sencilla en prueba. USÉ IA PARA ESTO, NO LO HICE YO
# El intercepto puede corresponder a un estudiante con valores que no existen en la muestra, por lo que no siempre tiene una interpretación práctica. USÉ IA PARA ESTO, NO LO HICE YO

# ## 9. Regresión logística
# 
# El modelo estima la probabilidad de colocación laboral y clasifica como colocado cuando esa probabilidad es igual o superior a 0,50. USÉ IA PARA ESTO, NO LO HICE YO
# Se utiliza la configuración habitual de scikit-learn, con regularización L2 para moderar los coeficientes, sin buscar hiperparámetros sobre el conjunto de prueba. USÉ IA PARA ESTO, NO LO HICE YO
# La probabilidad es una estimación del modelo y no una garantía de colocación ni un intervalo de confianza. USÉ IA PARA ESTO, NO LO HICE YO

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

# Accuracy representa la proporción total de clasificaciones correctas y debe compararse con la estrategia de predecir siempre la clase más frecuente. USÉ IA PARA ESTO, NO LO HICE YO
# Precision indica cuántos estudiantes realmente estaban colocados entre quienes el modelo clasificó como colocados; una precisión baja podría generar expectativas demasiado optimistas. USÉ IA PARA ESTO, NO LO HICE YO
# Recall indica cuántos de los estudiantes realmente colocados fueron identificados; un recall bajo dejaría sin reconocer parte de esos casos. USÉ IA PARA ESTO, NO LO HICE YO
# En este trabajo la clase positiva es colocado; identificar estudiantes en riesgo de no colocarse requiere considerar especialmente los errores de la fila No colocado en la matriz. USÉ IA PARA ESTO, NO LO HICE YO
# Si un modelo no predice ningún colocado, precision no está definida y el código muestra cero por convención, no como una medición ordinaria de precisión. USÉ IA PARA ESTO, NO LO HICE YO
# Un odds ratio mayor que uno representa mayores odds de colocación y uno menor que uno representa menores odds, manteniendo constantes las otras variables. USÉ IA PARA ESTO, NO LO HICE YO
# Las odds corresponden a probabilidad dividida por uno menos probabilidad; por ejemplo, un odds ratio de 1,20 significa un aumento de 20 % en las odds, no de 20 puntos porcentuales en la probabilidad. USÉ IA PARA ESTO, NO LO HICE YO
# Se divide cada coeficiente estandarizado por la escala de su variable para expresar el odds ratio por una hora, tarea, punto o punto porcentual original. USÉ IA PARA ESTO, NO LO HICE YO

# ## 10. Revisión sencilla de supuestos y variables relevantes
# 
# Los gráficos de relaciones anteriores y el gráfico de residuos ayudan a detectar patrones incompatibles con una relación lineal sencilla. USÉ IA PARA ESTO, NO LO HICE YO
# Un residuo es el puntaje observado menos el puntaje estimado; un patrón curvo o una dispersión que cambia mucho merece discusión. USÉ IA PARA ESTO, NO LO HICE YO
# La independencia no puede demostrarse con estos gráficos: debemos revisar el origen del dataset y si hay estudiantes repetidos, grupos o registros relacionados. USÉ IA PARA ESTO, NO LO HICE YO
# La regresión logística supone una relación lineal entre los predictores continuos y el logaritmo de las odds, no entre los predictores y la probabilidad directamente. USÉ IA PARA ESTO, NO LO HICE YO
# Este notebook no verifica formalmente ese último supuesto ni establece causalidad; lo declara como una limitación del análisis introductorio. USÉ IA PARA ESTO, NO LO HICE YO

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

# Las columnas por una desviación estándar permiten comparar magnitudes dentro de cada modelo sin confundir diferencias de escala entre horas y porcentajes. USÉ IA PARA ESTO, NO LO HICE YO
# Este orden de coeficientes es descriptivo, puede verse afectado por predictores relacionados y no demuestra importancia causal ni significancia estadística. USÉ IA PARA ESTO, NO LO HICE YO
# No se comparan numéricamente el efecto lineal y el logístico entre sí porque tienen unidades y objetivos diferentes. USÉ IA PARA ESTO, NO LO HICE YO
# Interpretación por completar: indicar qué variables muestran los efectos más grandes en cada modelo y si sus signos respaldan o contradicen la hipótesis principal. USÉ IA PARA ESTO, NO LO HICE YO

# ## 11. Predicciones para estudiantes no utilizados en entrenamiento
# 
# Se muestran diez casos de prueba con sus resultados reales y predichos; la fila original solo sirve para identificar el registro y no se utiliza como predictor. USÉ IA PARA ESTO, NO LO HICE YO
# Estos casos representan estudiantes no vistos, pero no prueban por sí solos que el modelo anticipará tendencias en años futuros. USÉ IA PARA ESTO, NO LO HICE YO

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

# Aplicación propuesta: usar el puntaje estimado como una señal complementaria para priorizar tutorías, considerando la magnitud de los errores observados. USÉ IA PARA ESTO, NO LO HICE YO
# Aplicación propuesta: utilizar la probabilidad de colocación como apoyo para orientar preparación laboral y revisar los errores antes de asignar recursos. USÉ IA PARA ESTO, NO LO HICE YO
# Las decisiones deben incorporar información adicional y revisión humana, porque estos modelos pueden equivocarse y los datos podrían no representar a otra institución. USÉ IA PARA ESTO, NO LO HICE YO

# ## 12. Comparación, discusión y conclusiones
# 
# Completar esta sección después de ejecutar el notebook; no interpretar MAE, RMSE o accuracy mediante umbrales arbitrarios de bueno o malo. USÉ IA PARA ESTO, NO LO HICE YO
# - Resultado académico: el MAE fue [valor] puntos y el RMSE fue [valor] puntos, frente a [valores] del modelo de referencia. USÉ IA PARA ESTO, NO LO HICE YO
# - El R cuadrado fue [valor] y su interpretación para esta muestra es [explicación]. USÉ IA PARA ESTO, NO LO HICE YO
# - Resultado laboral: accuracy fue [valor], precision fue [valor] y recall fue [valor], frente a [valores] de la clase más frecuente. USÉ IA PARA ESTO, NO LO HICE YO
# - Desde la gestión educativa, el error más relevante de la clasificación fue [tipo de error] porque [consecuencia práctica]. USÉ IA PARA ESTO, NO LO HICE YO
# - Las variables con mayores efectos descriptivos fueron [variables] en el modelo académico y [variables] en el laboral. USÉ IA PARA ESTO, NO LO HICE YO
# - La hipótesis principal recibió [apoyo, apoyo parcial o falta de apoyo] porque [evidencia de los coeficientes y resultados]. USÉ IA PARA ESTO, NO LO HICE YO
# - La comparación exploratoria sobre sueño mostró [patrón o ausencia de patrón], con [cantidad] estudiantes en cada grupo relevante. USÉ IA PARA ESTO, NO LO HICE YO
# 
# Los modelos no se comparan usando directamente R cuadrado y accuracy: se compara su utilidad en sus respectivas tareas y su mejora frente a referencias sencillas. USÉ IA PARA ESTO, NO LO HICE YO
# 
# Resumen de tres hallazgos respaldados por salidas: USÉ IA PARA ESTO, NO LO HICE YO
# 1. [Hallazgo académico y cifra que lo respalda]. USÉ IA PARA ESTO, NO LO HICE YO
# 2. [Hallazgo de colocación laboral y cifra que lo respalda]. USÉ IA PARA ESTO, NO LO HICE YO
# 3. [Hallazgo sobre variables o comparación de estudio y sueño]. USÉ IA PARA ESTO, NO LO HICE YO
# 
# Limitaciones: no se verificó formalmente la independencia ni la linealidad del logaritmo de las odds, no se evaluó otra institución o período y el análisis de grupos no controla otras características. USÉ IA PARA ESTO, NO LO HICE YO
# Revisar la documentación del dataset para confirmar si es sintético o basado en estudiantes reales y describir su origen antes de atribuirle representatividad empresarial. USÉ IA PARA ESTO, NO LO HICE YO
# La probabilidad logística no fue sometida a un análisis de calibración y no debe interpretarse como certeza individual. USÉ IA PARA ESTO, NO LO HICE YO

# ## 13. Guion sugerido de presentación y entrega
# 
# Preparar diapositivas con los resultados reales y exportar la presentación a PDF, manteniendo una duración total aproximada de doce minutos. USÉ IA PARA ESTO, NO LO HICE YO
# 1. Integrantes, contexto, objetivos e hipótesis: un minuto y medio. USÉ IA PARA ESTO, NO LO HICE YO
# 2. Dataset, exploración básica y limpieza: dos minutos. USÉ IA PARA ESTO, NO LO HICE YO
# 3. Variables, división y estandarización: un minuto. USÉ IA PARA ESTO, NO LO HICE YO
# 4. Regresión lineal y sus métricas: dos minutos. USÉ IA PARA ESTO, NO LO HICE YO
# 5. Regresión logística, odds ratios y matriz de confusión: dos minutos. USÉ IA PARA ESTO, NO LO HICE YO
# 6. Supuestos e hipótesis adicional: un minuto y medio. USÉ IA PARA ESTO, NO LO HICE YO
# 7. Comparación, tres hallazgos, limitaciones y aplicación: dos minutos. USÉ IA PARA ESTO, NO LO HICE YO
# 
# Antes de entregar, reemplazar todos los textos entre corchetes, incluir nombres de integrantes, comprobar las salidas de cada celda y conservar la declaración de uso de IA exigida por la pauta también en la presentación. USÉ IA PARA ESTO, NO LO HICE YO
# 
# Fuentes: pauta Regular_02.pdf y dataset de shambhurajejagadale en https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset. USÉ IA PARA ESTO, NO LO HICE YO

