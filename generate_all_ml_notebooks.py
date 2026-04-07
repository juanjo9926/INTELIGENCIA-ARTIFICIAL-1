# -*- coding: utf-8 -*-
"""
Genera los 9 notebooks de ML completos (secciones 0-8) en cada carpeta de dataset.
Ejecutar: python generate_all_ml_notebooks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebook_helpers import find_data_path, project_root

ROOT = project_root()


def save_nb(path: Path, cells: list) -> None:
    """Guarda un notebook en disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook()
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata["kernelspec"].update(
        display_name="Python 3", language="python", name="python3"
    )
    nb.cells = cells
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def md(text: str):
    """Celda markdown."""
    return new_markdown_cell(text)


def code(text: str):
    """Celda de codigo."""
    return new_code_cell(text)


# ---------------------------------------------------------------------------
# Dataset 1: Ames Housing
# ---------------------------------------------------------------------------
def build_ames_housing() -> list:
    """Notebook regresion: SalePrice."""
    p_train = find_data_path("1. Ames Housing Dataset", "Ames Housing", "train.csv")
    return [
        md(
            r"""# Pipeline ML — Ames Housing (Iowa, USA)

## SECCIÓN 0 — DESCRIPCIÓN DEL DATASET

### Origen y contexto
- **Origen**: conjunto de datos de viviendas de **Ames, Iowa (EE.UU.)**, usado habitualmente en competiciones (p. ej. enfoque tipo Kaggle).
- **Creador / difusión**: derivado de registros públicos de ventas y características de viviendas; no es un censo oficial sino un **dataset tabular** para aprendizaje automático.
- **Qué mide**: cada fila es una **venta de vivienda** con decenas de variables (tamaño del lote, calidad, barrio, año de construcción, etc.) y el **precio de venta**.

### Tarea de machine learning
- **Tipo**: **regresión** (variable objetivo continua).
- **Variable objetivo `y`**: `SalePrice` — precio de venta en **dólares estadounidenses (USD)**.
- **Uso típico**: predecir el precio a partir de las características de la casa (modelos lineales, bosques aleatorios, redes neuronales, etc.).

Las siguientes celdas cargan los datos y muestran forma, tipos y estadísticos descriptivos."""
        ),
        code(
            rf"""# Importaciones necesarias para todo el notebook
import warnings  # Suprimir avisos repetitivos de librerias
warnings.filterwarnings("ignore")  # Ignorar warnings no criticos

from pathlib import Path  # Rutas de archivos portables

import numpy as np  # Algebra lineal y arrays
import pandas as pd  # Tablas (DataFrames)
import matplotlib.pyplot as plt  # Graficos base
import seaborn as sns  # Graficos estadisticos avanzados
from IPython.display import display  # Mostrar tablas en Jupyter

from sklearn.model_selection import train_test_split  # Particion train/test
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Escalado y codificacion
from sklearn.linear_model import LinearRegression  # Modelo base de regresion
from sklearn.metrics import mean_squared_error, r2_score  # Metricas de regresion

# Ruta absoluta al CSV de entrenamiento (robusta ante distintos cwd)
ROOT_NB = Path(r"{str(ROOT).replace(chr(92), '/')}")  # Raiz del proyecto
CSV_PATH = ROOT_NB / "1. Ames Housing Dataset" / "Ames Housing" / "train.csv"  # Archivo de datos

df = pd.read_csv(CSV_PATH)  # Cargar datos en memoria

print("=== df.head(10) ===")  # Titulo de salida
display(df.head(10))  # Primeras 10 filas (en Jupyter muestra tabla HTML)

print("=== df.shape ===")  # Dimensiones
print(df.shape)  # (filas, columnas)

print("=== df.dtypes ===")  # Tipos por columna
print(df.dtypes)  # object/int64/float64/str segun pandas

print("=== df.describe() ===")  # Estadisticos numericos
display(df.describe(include="all").T.head(30))  # Resumen (truncado si es muy largo)
"""
        ),
        md(
            r"""## SECCIÓN 1 — SANEAMIENTO

**Decisiones generales:**
- **Mediana** para numéricos con faltantes: la mediana es **robusta a valores extremos**; la media se deforma si hay outliers (precios o superficies muy altos).
- **Moda** (valor más frecuente) para categóricas: preserva categorías observadas y es estándar con pocos faltantes.
- **Duplicados**: se eliminan filas idénticas para no contar el mismo hecho dos veces.
- **Columnas con >40% de nulos**: se eliminan porque aportan poca información y aumentan ruido/imputación poco fiable.
- **IDs** (`Order`, `PID`): se eliminan porque son identificadores, no causas del precio.

Tras el saneamiento, los nulos por columna deben ser **0**."""
        ),
        code(
            r"""df1 = df.copy()  # Copia para no modificar el original directamente

print("--- Nulos ANTES por columna (suma) ---")  # Etiqueta
print(df1.isnull().sum().sort_values(ascending=False).head(25))  # Conteo de nulos

# Duplicados: cuantas filas repetidas hay
n_dup = df1.duplicated().sum()  # Numero de filas duplicadas
df1 = df1.drop_duplicates()  # Eliminar duplicados (conservar primera aparicion)
print("Filas duplicadas eliminadas:", n_dup)  # Informar cuantas se quitaron

# Eliminar columnas con mas del 40% de valores nulos
null_ratio = df1.isnull().mean()  # Proporcion de nulos por columna
drop_high_null = null_ratio[null_ratio > 0.40].index.tolist()  # Lista de columnas a borrar
print("Columnas eliminadas por >40% nulos:", drop_high_null)  # Motivo: demasiada informacion faltante
df1 = df1.drop(columns=drop_high_null, errors="ignore")  # Borrar esas columnas

# Eliminar identificadores irrelevantes para el modelo
drop_ids = [c for c in ("Order", "PID") if c in df1.columns]  # IDs de fila/propiedad
print("Columnas ID eliminadas:", drop_ids)  # Explicacion: no son predictores causales
df1 = df1.drop(columns=drop_ids, errors="ignore")  # Quitar IDs

# Imputacion: numericas -> mediana, categoricas (object/str) -> moda
num_cols = df1.select_dtypes(include=[np.number]).columns.tolist()  # Columnas numericas
cat_cols = [c for c in df1.columns if c not in num_cols]  # Resto = categoricas/texto

for col in num_cols:  # Recorrer numericas
    med = df1[col].median()  # Mediana de la columna
    df1[col] = df1[col].fillna(med)  # Rellenar NaN con mediana

for col in cat_cols:  # Recorrer categoricas
    if df1[col].isnull().any():  # Solo si hay nulos
        moda = df1[col].mode(dropna=True)  # Moda (puede haber empate)
        valor = moda.iloc[0] if len(moda) > 0 else ""  # Primer valor modal
        df1[col] = df1[col].fillna(valor)  # Rellenar con moda

print("--- Nulos DESPUES por columna ---")  # Verificacion final
print(df1.isnull().sum().sum())  # Suma total de nulos (debe ser 0)
assert df1.isnull().sum().sum() == 0, "Aun quedan nulos"  # Comprobar que no queden nulos

df_clean = df1.copy()  # DataFrame saneado para siguientes secciones
"""
        ),
        md(
            r"""### Columnas conservadas vs eliminadas (resumen)

- **Eliminadas**: columnas con **>40% nulos** (listadas en la salida anterior) y **Order/PID** (identificadores).
- **Conservadas**: el resto aporta información sobre ubicación, calidad, tamaño y estado de la vivienda, útil para predecir precio.

## SECCIÓN 2 — IDENTIFICACIÓN DE X e y

- **y**: `SalePrice` (precio a predecir).
- **X**: todas las demás columnas tras el saneamiento.

Para regresión, mostramos la **distribución del precio** (histograma) en lugar de `value_counts` de clases. Las **correlaciones** con el precio ayudan a ver qué variables numéricas se mueven con él."""
        ),
        code(
            r"""y = df_clean["SalePrice"].astype(float)  # Objetivo: precio de venta
X = df_clean.drop(columns=["SalePrice"])  # Features: todo excepto el precio

print("X.shape =", X.shape)  # Filas y columnas de features
print("y.shape =", y.shape)  # Longitud del vector objetivo

# Correlacion solo con columnas numericas en X
num_x = X.select_dtypes(include=[np.number]).columns  # Subconjunto numerico
corr_with_price = X[num_x].corrwith(y).abs().sort_values(ascending=False)  # |corr| con y
print("Top 5 features mas correlacionadas (en valor absoluto) con SalePrice:")  # Titulo
print(corr_with_price.head(5))  # Cinco mayores correlaciones

plt.figure(figsize=(8, 4))  # Tamanio de figura
plt.hist(y, bins=40, color="steelblue", edgecolor="black")  # Histograma de precios
plt.xlabel("SalePrice (USD)")  # Etiqueta eje X
plt.ylabel("Frecuencia")  # Etiqueta eje Y
plt.title("Distribucion del precio de venta (regresion)")  # Titulo del grafico
plt.tight_layout()  # Ajustar margenes
plt.show()  # Mostrar grafico

# Mapa de calor pequeno: top numericas + precio
top5 = corr_with_price.head(5).index.tolist()  # Nombres de las 5 variables
heat_df = df_clean[top5 + ["SalePrice"]].corr()  # Matriz de correlacion
plt.figure(figsize=(7, 5))  # Tamanio
sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="coolwarm")  # Heatmap
plt.title("Correlaciones entre top 5 numericas y SalePrice")  # Titulo
plt.tight_layout()  # Ajuste
plt.show()  # Mostrar
"""
        ),
        md(
            r"""## SECCIÓN 3 — ENCODING

- **Nominales** (sin orden intrínseco): barrio (`Neighborhood`), tipo de zona (`MS Zoning`), etc. → **one-hot** (`pd.get_dummies`) para no inventar orden.
- **Ordinales** (escala de calidad): p. ej. `Exter Qual`, `Kitchen Qual`, `Bsmt Qual` (típicamente Ex > Gd > TA > Fa > Po) → **LabelEncoder** con orden fijo cuando es posible.

Listamos columnas categóricas (no numéricas) y aplicamos la estrategia indicada."""
        ),
        code(
            r"""X_enc = X.copy()  # Copia para codificar

# Detectar columnas no numericas (object o string en pandas 2+)
non_num = [c for c in X_enc.columns if not pd.api.types.is_numeric_dtype(X_enc[c])]  # Lista categoricas

# Lista nominal vs ordinal por nombre (heuristica documentada)
ordinal_keywords = ("Qual", "QC", "Cond", "Qu")  # Subcadenas que sugieren escala ordinal
ordinal_cols = []  # Lista de ordinales detectadas
nominal_cols = []  # Lista de nominales

for col in non_num:  # Clasificar cada columna categorica
    if any(k in col for k in ordinal_keywords):  # Si el nombre sugiere calidad/condicion
        ordinal_cols.append(col)  # Tratar como ordinal
    else:
        nominal_cols.append(col)  # Tratar como nominal

print("Columnas ordinales (LabelEncoder):", ordinal_cols)  # Mostrar lista
print("Columnas nominales (one-hot):", nominal_cols)  # Mostrar lista

# Label encoding para ordinales: mapear orden Ex>Gd>TA>Fa>Po si aparecen
orden_qual = ["Ex", "Gd", "TA", "Fa", "Po", "NA", "None", ""]  # Orden de mejor a peor aproximado

for col in ordinal_cols:  # Codificar cada ordinal
    le = LabelEncoder()  # Codificador de etiquetas
    # Convertir a string y reemplazar vacios
    s = X_enc[col].astype(str).replace("nan", np.nan).fillna("NA")  # Texto limpio
    # Si los valores son subconjunto de niveles conocidos, usar factorize por orden
    uniq = sorted(s.unique().tolist())  # Valores unicos
    try:
        X_enc[col] = le.fit_transform(s)  # Etiquetas 0..k-1
    except Exception:
        X_enc[col] = pd.Categorical(s).codes  # Respaldo: codigos de categoria

# One-hot para nominales
X_enc = pd.get_dummies(X_enc, columns=nominal_cols, drop_first=False)  # k dummies por nominal

print("Columnas ANTES encoding (aprox):", list(X.columns)[:15], "...")  # Muestra previa
print("Columnas DESPUES encoding:", X_enc.shape[1])  # Numero total de columnas nuevas
display(X_enc.head(5))  # Primeras 5 filas codificadas
"""
        ),
        md(
            r"""## SECCIÓN 4 — NORMALIZACIÓN

**StandardScaler**: transforma cada columna a media **0** y varianza **1** usando \(z = (x - \mu) / \sigma\) **calculados en el conjunto que se usa para fit** (aquí, todo el dataset antes del split; en producción se recomienda fit solo en train).

**Por qué importa en redes neuronales**: las unidades reciben inputs en escalas similares, evitando que unas dimensiones dominen el entrenamiento (gradientes más estables).

Mostramos **5 filas** antes y después en un subconjunto de columnas numéricas."""
        ),
        code(
            r"""# Guardar copia antes de escalar (para comparacion)
antes = X_enc.iloc[:5].copy()  # Cinco primeras filas antes del escalado

scaler = StandardScaler()  # Instancia de escalado estandar
X_scaled = scaler.fit_transform(X_enc)  # Matriz numpy normalizada
X_scaled_df = pd.DataFrame(X_scaled, columns=X_enc.columns, index=X_enc.index)  # Volver a DataFrame

despues = X_scaled_df.iloc[:5].copy()  # Cinco filas despues

print("=== Comparativa 5 filas: ANTES (izquierda) vs DESPUES (derecha) [primeras 8 columnas] ===")  # Titulo
cols_show = list(X_enc.columns[:8])  # Subconjunto de columnas para legibilidad
comp = pd.concat([antes[cols_show], despues[cols_show]], axis=1, keys=["ANTES", "DESPUES"])  # Tabla unida
display(comp)  # Mostrar comparativa
"""
        ),
        md(
            r"""## SECCIÓN 5 — TRAIN / TEST SPLIT

- **80% entrenamiento / 20% prueba**, `random_state=42` para **reproducibilidad**.
- **Train**: ajuste del modelo; **test**: estimación honesta del error fuera de muestra."""
        ),
        code(
            r"""X_train, X_test, y_train, y_test = train_test_split(  # Particion estratificada no aplica a regresion continua
    X_scaled_df,  # Features ya escaladas
    y,  # Objetivo
    test_size=0.2,  # 20% para test
    random_state=42,  # Semilla fija
)

print("X_train.shape =", X_train.shape)  # Tamano train
print("X_test.shape =", X_test.shape)  # Tamano test
print("y_train.shape =", y_train.shape)  # Tamano y train
print("y_test.shape =", y_test.shape)  # Tamano y test
"""
        ),
        md(
            r"""## SECCIÓN 6 — MODELO DE PRUEBA (baseline)

**LinearRegression** como comprobación de que el pipeline funciona. Métricas:
- **MSE**: error cuadrático medio (penaliza grandes errores).
- **R²**: fracción de varianza explicada (1 es perfecto, 0 similar a predecir la media).

No es el modelo final; sirve de **sanity check**."""
        ),
        code(
            r"""modelo = LinearRegression()  # Regresion lineal ordinaria
modelo.fit(X_train, y_train)  # Entrenar con train
pred = modelo.predict(X_test)  # Predecir en test

mse = mean_squared_error(y_test, pred)  # Error cuadratico medio
r2 = r2_score(y_test, pred)  # Coeficiente de determinacion

print("MSE en test:", round(mse, 2))  # Imprimir MSE
print("R2 en test:", round(r2, 4))  # Imprimir R2
print("Interpretacion: R2 cercano a 1 indica buen ajuste; MSE en unidades USD^2.")  # Explicacion breve
"""
        ),
        md(
            r"""## SECCIÓN 7 — GUARDADO DE ARCHIVOS

Se guardan el CSV limpio (sin escalar, con `SalePrice` para referencia) y los cuatro CSV de split **con features escaladas** para entrenamiento directo."""
        ),
        code(
            r"""out_dir = ROOT_NB / "1. Ames Housing Dataset"  # Carpeta del dataset
out_dir.mkdir(parents=True, exist_ok=True)  # Crear si no existe

# Tabla final lista para modelado: features escaladas + objetivo
df_final_ml = pd.concat([X_scaled_df, y.rename("SalePrice")], axis=1)  # Unir X normalizado y precio
fname_clean = out_dir / "ames_housing_clean.csv"  # Archivo de dataset limpio final
df_final_ml.to_csv(fname_clean, index=False)  # Guardar matriz final
print("Guardado:", fname_clean)  # Confirmacion

X_train.to_csv(out_dir / "X_train.csv", index=False)  # Train features escaladas
X_test.to_csv(out_dir / "X_test.csv", index=False)  # Test features escaladas
y_train.to_csv(out_dir / "y_train.csv", index=True, header=True)  # Train objetivo
y_test.to_csv(out_dir / "y_test.csv", index=True, header=True)  # Test objetivo
print("Guardados: X_train.csv, X_test.csv, y_train.csv, y_test.csv")  # Confirmacion
"""
        ),
        md(
            r"""## SECCIÓN 8 — RESUMEN FINAL

Comprobaciones finales y caja de resumen."""
        ),
        code(
            r"""display(df_clean.head(10))  # Diez primeras filas del dataframe saneado (con SalePrice)

print("Shape final df_clean:", df_clean.shape)  # Dimensiones
print("Columnas finales (lista):", list(df_clean.columns))  # Nombres

assert df_clean.isnull().sum().sum() == 0, "Hay nulos"  # Sin nulos
# Tras encoding en X_scaled_df todo es numerico
assert np.all(np.isfinite(X_scaled_df.values)), "Hay no finitos"  # Sin inf/NaN en X escalado

print("+------------- RESUMEN -------------+")  # Caja resumen
print("| Dataset: Ames Housing (Iowa)")  # Nombre
print("| Objetivo: precio SalePrice (USD)")  # Target
print("| Filas/columnas finales (saneado):", df_clean.shape)  # Shape
print("| Numero de features en X tras encoding:", X_enc.shape[1])  # Features
print("| Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])  # Particion
print("| Baseline R2 (test):", round(r2, 4))  # Metrica
print("+-----------------------------------+")  # Cierre
"""
        ),
    ]


def main() -> None:
    """Genera todos los notebooks (placeholder: solo Ames en primera version)."""
    path = ROOT / "1. Ames Housing Dataset" / "ames_housing_ml.ipynb"
    save_nb(path, build_ames_housing())
    print("Generado:", path)


if __name__ == "__main__":
    main()
