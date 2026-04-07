# -*- coding: utf-8 -*-
"""Genera notebooks ML para datasets 2-9. Ejecutar tras generate_all_ml_notebooks.py (Ames)."""
from __future__ import annotations

import warnings
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebook_helpers import project_root

ROOT = project_root()
warnings.filterwarnings("ignore")


def save_nb(path: Path, cells: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook()
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata["kernelspec"].update(
        display_name="Python 3", language="python", name="python3"
    )
    nb.cells = cells
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def md(t: str):
    return new_markdown_cell(t)


def code(t: str):
    return new_code_cell(t)


# Raiz embebida para notebooks (se reemplaza al generar)
ROOT_STR = str(ROOT).replace("\\", "/")


def imports_cell() -> str:
    return f"""import warnings  # Silenciar avisos
warnings.filterwarnings("ignore")  # Ignorar warnings
from pathlib import Path  # Rutas
import numpy as np  # Numerico
import pandas as pd  # DataFrames
import matplotlib.pyplot as plt  # Graficos
import seaborn as sns  # Graficos stats
from IPython.display import display  # Tablas Jupyter
from sklearn.model_selection import train_test_split  # Split
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Preprocesamiento
from sklearn.linear_model import LinearRegression, LogisticRegression  # Modelos
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score  # Metricas
ROOT_NB = Path(r"{ROOT_STR}")  # Raiz proyecto primerparcial_ia
"""


def build_mimic() -> list:
    """MIMIC-III: ADMISSIONS unido a PATIENTS; objetivo hospital_expire_flag."""
    return [
        md(
            r"""# MIMIC-III Demo — Clasificación (fallecimiento hospitalario)

## SECCIÓN 0 — DESCRIPCIÓN DEL DATASET

- **Origen**: **MIMIC-III** (Medical Information Mart for Intensive Care), datos de la **Beth Israel Deaconess Medical Center** (Boston); versión **demo** reducida.
- **Contexto real**: admisiones a UCI/hospital con diagnósticos, tiempos y desenlaces.
- **Tarea ML**: **clasificación binaria**.
- **Variable objetivo `y`**: `hospital_expire_flag` (**0** = alta viva, **1** = fallecimiento durante la estancia hospitalaria según el registro).

Unimos **ADMISSIONS** con **PATIENTS** por `subject_id` para añadir sexo y edad aproximada."""
        ),
        code(
            imports_cell()
            + f"""
mimic_root = list(ROOT_NB.glob("2. MIMIC*"))[0]  # Carpeta MIMIC con nombre unicode
mimic_data = mimic_root / "mimic-iii-clinical-database-demo-1.4"  # Subcarpeta datos
adm = pd.read_csv(mimic_data / "ADMISSIONS.csv")  # Tabla admisiones
pat = pd.read_csv(mimic_data / "PATIENTS.csv")  # Tabla pacientes
df = adm.merge(pat, on="subject_id", how="left", suffixes=("", "_pat"))  # Unir por paciente
print("Forma tras merge:", df.shape)  # Dimension
display(df.head(10))  # Muestra
print("shape:", df.shape)  # Repetir shape
print("dtypes:\\n", df.dtypes)  # Tipos
display(df.describe(include="all").T.head(25))  # Describe
"""
        ),
        md(
            r"""## SECCIÓN 1 — SANEAMIENTO

- **Mediana** en numéricos: robustez a outliers (p. ej. tiempos).
- **Moda** en categóricas: valor más frecuente.
- **Eliminar duplicados** de filas.
- **Eliminar columnas** con >40% nulos o identificadores (`row_id`, `hadm_id`, `subject_id`) que no deben usarse como predictores directos (riesgo de fuga de información o sobreajuste espurio).
- **Fechas**: convertir a numérico (timestamp o año) para modelos."""
        ),
        code(
            f"""df1 = df.copy()  # Copia
print("Nulos ANTES:\\n", df1.isnull().sum().sort_values(ascending=False).head(20))  # Nulos
df1 = df1.drop_duplicates()  # Quitar duplicados
null_ratio = df1.isnull().mean()  # Ratio nulos
df1 = df1.drop(columns=null_ratio[null_ratio > 0.40].index.tolist(), errors="ignore")  # >40% nulos
drop_ids = [c for c in ("row_id", "hadm_id", "subject_id") if c in df1.columns]  # IDs
print("Columnas ID eliminadas (no predictores causales directos):", drop_ids)  # Log
df1 = df1.drop(columns=drop_ids, errors="ignore")  # Borrar IDs
# Fechas a epoch segundos (simplificado)
for col in list(df1.columns):
    if "time" in col.lower() or col in ("dob", "dod", "dod_hosp", "dod_ssn", "admittime", "dischtime", "deathtime", "edregtime", "edouttime"):
        if col in df1.columns:
            df1[col] = pd.to_datetime(df1[col], errors="coerce")  # Parseo fecha
            df1[col] = df1[col].astype("int64", errors="ignore") // 10**9  # Segundos epoch aprox
num_cols = df1.select_dtypes(include=[np.number]).columns.tolist()  # Numericas
cat_cols = [c for c in df1.columns if c not in num_cols]  # Categoricas
for col in num_cols:
    df1[col] = df1[col].fillna(df1[col].median())  # Mediana
for col in cat_cols:
    if df1[col].isnull().any():
        m = df1[col].mode()
        df1[col] = df1[col].fillna(m.iloc[0] if len(m) else "")  # Moda
print("Nulos DESPUES (suma total):", df1.isnull().sum().sum())  # Debe ser 0
assert df1.isnull().sum().sum() == 0
df_clean = df1.copy()
"""
        ),
        md(
            r"""## SECCIÓN 2 — X e y

- **y** = `hospital_expire_flag`
- **X** = resto de columnas (sin la objetivo).

Gráfico de barras de clases; correlación con la etiqueta codificada numéricamente para features numéricas."""
        ),
        code(
            """y = df_clean["hospital_expire_flag"].astype(int)  # Objetivo binario
X = df_clean.drop(columns=["hospital_expire_flag"])  # Features
print("X.shape", X.shape, "y.shape", y.shape)  # Shapes
print(y.value_counts())  # Distribucion clases
plt.figure(figsize=(5,3))  # Figura
y.value_counts().plot(kind="bar", color=["green","red"])  # Barras
plt.title("Distribucion hospital_expire_flag")  # Titulo
plt.xticks(rotation=0)  # Etiquetas
plt.show()  # Mostrar
imb = y.value_counts(normalize=True)  # Proporciones
print("Proporciones:", imb)  # Imbalance
if imb.min() < 0.2:
    print("Clases desbalanceadas: una clase es rara; implica metricas como F1 y estrategias de balanceo.")  # Nota
# Correlacion con objetivo (numerico)
num_x = X.select_dtypes(include=[np.number]).columns  # Num features
if len(num_x) > 0:
    corr = X[num_x].corrwith(y).abs().sort_values(ascending=False)  # Corr
    print("Top 5 correlaciones con y:", corr.head(5))  # Top 5
    sns.heatmap(pd.concat([X[num_x[:min(5,len(num_x))]], y], axis=1).corr(), annot=True)  # Heatmap pequeno
    plt.show()
"""
        ),
        md(
            r"""## SECCIÓN 3 — ENCODING

- **Nominales** (p. ej. tipo de admisión, seguro): **one-hot**.
- **Ordinales** si las hubiera con orden claro: **LabelEncoder**; aquí muchas categóricas son nominales."""
        ),
        code(
            """X_enc = X.copy()  # Copia
non_num = [c for c in X_enc.columns if not pd.api.types.is_numeric_dtype(X_enc[c])]  # No numericas
print("Categoricas antes:", non_num)  # Lista
X_enc = pd.get_dummies(X_enc, columns=non_num, drop_first=False)  # One-hot todo (nominal predominante)
print("Num columnas despues encoding:", X_enc.shape[1])  # Conteo
display(X_enc.head(5))  # Muestra
"""
        ),
        md(
            r"""## SECCIÓN 4 — NORMALIZACIÓN (StandardScaler)

Comparación 5 filas antes/después en primeras columnas."""
        ),
        code(
            """scaler = StandardScaler()  # Escalador
X_scaled = scaler.fit_transform(X_enc)  # Ajuste y transformacion
X_scaled_df = pd.DataFrame(X_scaled, columns=X_enc.columns, index=X_enc.index)  # DataFrame
antes = X_enc.iloc[:5, :6] if X_enc.shape[1] >= 6 else X_enc.iloc[:5]  # Antes (subcolumnas)
despues = X_scaled_df.iloc[:5, :6] if X_scaled_df.shape[1] >= 6 else X_scaled_df.iloc[:5]  # Despues
display(pd.concat([antes, despues], axis=1, keys=["ANTES","DESPUES"]))  # Comparativa
"""
        ),
        md(
            r"""## SECCIÓN 5 — TRAIN/TEST (80/20, random_state=42)"""
        ),
        code(
            """X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)  # Estratificar clases
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # Shapes
"""
        ),
        md(
            r"""## SECCIÓN 6 — Baseline: LogisticRegression(max_iter=1000)"""
        ),
        code(
            """clf = LogisticRegression(max_iter=1000)  # Modelo lineal clasico
clf.fit(X_train, y_train)  # Entrenar
acc = accuracy_score(y_test, clf.predict(X_test))  # Precision en test
print("Accuracy test:", round(acc, 4))  # Metrica
print("Accuracy = proporcion de aciertos; 1.0 es perfecto.")  # Significado
r2 = acc  # Guardar score para README
"""
        ),
        md(
            r"""## SECCIÓN 7 — GUARDADO CSV"""
        ),
        code(
            f"""out_dir = list(ROOT_NB.glob("2. MIMIC*"))[0]  # Carpeta dataset
df_final_ml = pd.concat([X_scaled_df, y.rename("hospital_expire_flag")], axis=1)  # Final
df_final_ml.to_csv(out_dir / "mimic_iii_clean.csv", index=False)  # Limpio
print("Guardado mimic_iii_clean.csv")  # Ok
X_train.to_csv(out_dir / "X_train.csv", index=False)  # Train X
X_test.to_csv(out_dir / "X_test.csv", index=False)  # Test X
y_train.to_csv(out_dir / "y_train.csv", index=True, header=True)  # y train
y_test.to_csv(out_dir / "y_test.csv", index=True, header=True)  # y test
print("Splits guardados")  # Ok
"""
        ),
        md(
            r"""## SECCIÓN 8 — RESUMEN Y ASSERTS"""
        ),
        code(
            """display(df_clean.head(10))  # Muestra
assert df_clean.isnull().sum().sum() == 0  # Sin nulos
assert np.all(np.isfinite(X_scaled_df.values))  # Finitos
print("+----------- RESUMEN -----------+")  # Caja
print("| MIMIC-III demo | Clasificacion | hospital_expire_flag |")  # Linea
print("| Test accuracy:", round(acc, 4), "|")  # Score
print("+--------------------------------+")  # Fin
"""
        ),
    ]


def main():
    p = ROOT / "2. MIMIC‑III (Medical Information Mart for Intensive Care)"
    p = list(ROOT.glob("2. MIMIC*"))[0]
    save_nb(p / "mimic_iii_ml.ipynb", build_mimic())
    print("OK MIMIC", p / "mimic_iii_ml.ipynb")


if __name__ == "__main__":
    main()
