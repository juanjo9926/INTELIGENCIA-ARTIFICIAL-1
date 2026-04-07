# -*- coding: utf-8 -*-
"""Regenera los cuadernos sanitizar_*.ipynb con explicaciones, antes/despues y tablas."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent

MD_PIPELINE = new_markdown_cell(
    r"""## Pipeline de `sklearn` (qué hace cada pieza)

| Pieza | Rol |
|-------|-----|
| **SimpleImputer (median)** | Sustituye NaN en numéricas por la mediana (robusto a outliers). |
| **StandardScaler** | Normaliza: \((x - \mu) / \sigma\) por columna → media ~0, varianza ~1. |
| **SimpleImputer (most_frequent)** | Sustituye NaN en categóricas por la categoría más frecuente. |
| **OneHotEncoder** | Convierte cada categoría en columnas 0/1 (una por valor visto al entrenar). |

La siguiente celda **define** `build_preprocessor`. La siguiente **ajusta** (`fit_transform`) y muestra **antes/después** en numéricas y una **muestra de la matriz final**."""
)

MD_CIERRE = new_markdown_cell(
    r"""## Cómo leer la salida del último bloque de código

| Sección | Qué verás |
|---------|-----------|
| **MUESTRA 10 filas de X** | Tus **features crudas** (antes de imputar/escalar/one-hot). Solo se muestran las primeras columnas para que quepa en pantalla. |
| **ANTES (normalización)** | Valores **originales** de las columnas **numéricas** (10 filas). |
| **DESPUÉS (normalización)** | Mismas columnas tras **StandardScaler**: cada columna tiene media ≈ 0 y desviación ≈ 1 **sobre todo el conjunto** (útil para redes y SVM). |
| **Resumen estadístico ANTES/DESPUÉS** | `mean`, `std`, `min`, `max` para comprobar el efecto del escalado. |
| **MATRIZ FINAL (10 × N)** | Matriz que **entra al modelo**: bloque numérico ya escalado + columnas **one-hot** de categorías (nombres tipo `cat__Sex_Male`). |
| **y** | Etiquetas: precio, clase, etc., según el dataset. |

**Nota:** Las columnas categóricas no aparecen en la tabla “ANTES/DESPUÉS” numérica; su información pasa a columnas 0/1 en la **MATRIZ FINAL**."""
)

SETUP = r'''import sys
from pathlib import Path

# --- Localizar la raiz del proyecto (carpeta donde estan .env y load_project_env.py) ---
def _project_root() -> Path:
    """Busca hacia arriba desde el directorio de trabajo hasta encontrar load_project_env.py."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "load_project_env.py").exists():
            return cand
    return p

ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_project_env import load_env, data_path

load_env()
print("PROJECT_ROOT =", ROOT)
'''

PIPELINE_FUN = r'''from sklearn.compose import ColumnTransformer  # Une transformaciones por tipo de columna
from sklearn.impute import SimpleImputer  # Rellena valores faltantes
from sklearn.pipeline import Pipeline  # Encadena pasos: imputer -> scaler / one-hot
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Escalado y codificacion categorica


def build_preprocessor(numeric_cols, categorical_cols):
    """Numericos: mediana + StandardScaler. Categoricos: moda + one-hot. Listas vacias = se omite ese bloque."""
    # Sub-pipeline numerico: primero imputar, luego llevar a escala comun
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # NaN -> mediana de la columna
            ("scaler", StandardScaler()),  # (x - media) / std por columna
        ]
    )
    # Sub-pipeline categorico: imputar texto y expandir a columnas binarias
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # NaN -> categoria mas frecuente
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),  # Una columna 0/1 por valor distinto visto al hacer fit
        ]
    )
    transformers = []  # Lista de tuplas (nombre, pipeline, lista de columnas)
    if numeric_cols:  # Si hay al menos una columna numerica
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:  # Si hay al menos una columna no numerica
        transformers.append(("cat", cat_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No hay columnas numericas ni categoricas para transformar.")
    # remainder='drop' ignora columnas no listadas (aqui no deberia haber)
    return ColumnTransformer(transformers=transformers, remainder="drop")
'''


def save(path: Path, cells: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook()
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata["kernelspec"].update(
        display_name="Python 3", language="python", name="python3"
    )
    nb.cells = cells
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def display_block() -> str:
    """Comparacion antes/despues de normalizar (numericas), matriz final, etiqueta y."""
    return r'''
# ========== PASO FINAL: entrenar transformaciones y obtener matriz para ML ==========
# fit_transform: aprende medianas/modas sobre X y devuelve la matriz ya transformada
prep = build_preprocessor(num_cols, cat_cols)  # Construye el objeto (aun no aprende estadisticos)
X_matrix = prep.fit_transform(X)  # Aqui se aprende y se aplica: salida 2D numpy

# Nombres de columnas de salida (prefijos num__ y cat__)
feature_names = list(prep.get_feature_names_out())
print("Shape X_matrix (listo para red neuronal / sklearn):", X_matrix.shape)
print("Total de features despues de one-hot:", len(feature_names))

pd.set_option("display.max_columns", 14)  # Limitar columnas al imprimir tablas
pd.set_option("display.width", 220)

# --- Comparar ANTES y DESPUES solo en columnas NUMERICAS (efecto de StandardScaler) ---
num_pipe_fitted = prep.named_transformers_.get("num")  # None si no hubo columnas numericas
if num_cols and num_pipe_fitted is not None:
    X_num_scaled = num_pipe_fitted.transform(X[num_cols])  # Mismas filas, solo bloque numerico escalado
    antes_num = X[num_cols].iloc[:10].copy()  # 10 filas tal cual en el CSV
    despues_num = pd.DataFrame(
        X_num_scaled[:10],
        columns=num_cols,
        index=antes_num.index,
    )
    print("\n=== NORMALIZACION (StandardScaler sobre numericas): ANTES — 10 filas, valores ORIGINALES ===")
    print(antes_num.to_string())
    print("\n=== NORMALIZACION: DESPUES — mismas 10 filas, ESCALADAS (media global ~0, std ~1 por columna) ===")
    print(despues_num.to_string())
    print("\n=== Resumen estadistico numericas ANTES (mean, std, min, max) ===")
    print(X[num_cols].describe().T[["mean", "std", "min", "max"]].head(20).to_string())
    print("\n=== Resumen estadistico numericas DESPUES del escalado (deberian tener mean~0, std~1) ===")
    print(pd.DataFrame(X_num_scaled, columns=num_cols).describe().T[["mean", "std", "min", "max"]].head(20).to_string())
else:
    print("\n(No hay columnas numericas en X o solo hay categoricas; no se aplica StandardScaler en bloque num.)")

# --- Salida mezclada: primeras columnas de la matriz final (numeros escalados + one-hot) ---
n_show = min(20, X_matrix.shape[1])
df_final_muestra = pd.DataFrame(
    X_matrix[:10, :n_show],
    columns=[str(feature_names[i])[:45] for i in range(n_show)],
)
print("\n=== MATRIZ FINAL: 10 filas x primeras " + str(n_show) + " columnas (nombres truncados; incluye one-hot) ===")
print(df_final_muestra.to_string())

# Etiqueta objetivo (regresion o clasificacion)
if y is None:
    print("\n[y] No hay vector objetivo en este cuaderno (solo features).")
else:
    print("\n[y] Forma del vector objetivo:", getattr(y, "shape", type(y)))
    print("[y] Primeras 10 etiquetas:", y[:10] if hasattr(y, "__getitem__") else y)
'''


def main() -> None:
    cells_ames = [
        new_markdown_cell(
            r"""# Ames Housing (Kaggle) — Regresión del precio de venta

## En qué consiste el dataset
- **Origen**: competición estilo "Ames Housing"; filas = casas vendidas, columnas = características del inmueble y venta.
- **Objetivo ML**: predecir **`SalePrice`** (regresión): precio de venta en dólares.
- **Por qué preprocesar**: mezcla de variables numéricas (metros, años) y categóricas (barrio, calidad). Los modelos (redes, regresión) necesitan números; las categóricas se codifican con **one-hot**; las numéricas se **imputan** (faltantes) y se **estandarizan** (misma escala).

## Qué hace este cuaderno (pasos)
1. Cargar `train.csv` desde la ruta del `.env`.
2. Separar **objetivo** `y` = `SalePrice` y **features** `X`.
3. **Eliminar** identificadores (`Order`, `PID`, `Id`) que no aportan patrón generalizable.
4. Clasificar columnas en **numéricas** vs **categóricas**.
5. Aplicar **ColumnTransformer**: mediana + `StandardScaler` en numéricas; moda + `OneHotEncoder` en categóricas.
6. Mostrar **antes/después** de normalizar (solo parte numérica) y **10 filas** de la matriz final."""
        ),
        new_code_cell(SETUP),
        new_code_cell(
            r'''import pandas as pd
import numpy as np

# Ruta definida en .env (DATA_AMES_TRAIN)
csv_path = data_path("DATA_AMES_TRAIN")
# Leemos todas las columnas del CSV
df_original = pd.read_csv(csv_path)

print("Filas y columnas al cargar:", df_original.shape)
print("Primeras columnas:", list(df_original.columns[:8]), "...")

# --- Objetivo: precio de venta (regresion) ---
target_col = "SalePrice"
y = df_original[target_col].astype(float).values

# --- Features: todo excepto el precio ---
X = df_original.drop(columns=[target_col])

# --- Columnas que ELIMINAMOS y por que ---
# Son identificadores unicos o de orden; no son causales del precio en generalization
drop_ids = [c for c in ("Id", "Order", "PID") if c in X.columns]
X = X.drop(columns=drop_ids)
print("\n[ELIMINADAS]", drop_ids if drop_ids else "(ninguna con esos nombres)")
print("Motivo: IDs no deben usarse como entrada (data leakage aparente / ruido).")

# --- CONSERVAMOS el resto: son atributos de la casa o del barrio ---
print("\n[CONSERVADAS]", X.shape[1], "columnas de entrada (features).")

# Tipos: numericas vs categoricas (object, string, category -> categoricas)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nResumen tipos:")
print("  Numericas:", len(num_cols), "| Categoricas:", len(cat_cols))
print("  Ejemplo numericas:", num_cols[:6])
print("  Ejemplo categoricas:", cat_cols[:6])

# Muestra legible: 10 filas y solo las primeras columnas (evita salida enorme)
n_preview = min(12, X.shape[1])
print("\n=== MUESTRA: 10 filas de X (primeras", n_preview, "columnas de", X.shape[1], ") ===")
print(X.iloc[:10, :n_preview].to_string())
'''
        ),
        MD_PIPELINE,
        new_code_cell(PIPELINE_FUN),
        new_code_cell(display_block()),
        MD_CIERRE,
    ]

    save(ROOT / "1. Ames Housing Dataset" / "sanitizar_ames_housing.ipynb", cells_ames)

    # MIMIC
    cells_mimic = [
        new_markdown_cell(
            r"""# MIMIC-III DEMO — Tabla PATIENTS

## En qué consiste
- **MIMIC-III** es una base clínica; la tabla **PATIENTS** tiene un paciente por fila: género, fechas de nacimiento/defunción, bandera de fallecimiento.
- **Objetivo ML propuesto**: predecir **`expire_flag`** (clasificación binaria: si figura fallecido en el registro).

## Qué hace este cuaderno
1. Cargar `PATIENTS.csv`.
2. Quitar **row_id** y **subject_id** (identificadores).
3. Convertir fechas a **número ordinal** (días desde una época) para que entren redes/regresiones sin texto.
4. One-hot en género; escalar ordinales.
5. Mostrar columnas finales, antes/después de escalar numéricos, y 10 filas."""
        ),
        new_code_cell(SETUP),
        new_code_cell(
            r'''import pandas as pd
import numpy as np

path = data_path("DATA_MIMIC_PATIENTS")
df_original = pd.read_csv(path)
print("Forma original:", df_original.shape)
print("Columnas:", list(df_original.columns))

# IDs fuera
drop_ids = [c for c in ("row_id", "subject_id") if c in df_original.columns]
df = df_original.drop(columns=drop_ids)
print("\n[ELIMINADAS]", drop_ids, "— identificadores de fila/paciente, no son señal clinica directa.")

# Fechas -> ordinal (conservamos informacion temporal como numero)
for col in ("dob", "dod", "dod_hosp", "dod_ssn"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col + "_ordinal"] = df[col].apply(
            lambda t: t.toordinal() if pd.notna(t) else np.nan
        )
        df = df.drop(columns=[col])
        print("[TRANSFORMADA]", col, "->", col + "_ordinal", "(fecha a entero ordinal)")

y = df["expire_flag"].astype(int).values
X = df.drop(columns=["expire_flag"])
print("\n[OBJETIVO] y = expire_flag (0/1). [CONSERVAMOS] features en X.")

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numericas:", num_cols, "| Categoricas:", cat_cols)

n_preview = min(12, X.shape[1])
print("\n=== MUESTRA: 10 filas de X (primeras", n_preview, "columnas) ===")
print(X.iloc[:10, :n_preview].to_string())
'''
        ),
        MD_PIPELINE,
        new_code_cell(PIPELINE_FUN),
        new_code_cell(display_block()),
        MD_CIERRE,
    ]
    mimic_dir = next(ROOT.glob("2. MIMIC*"))
    save(mimic_dir / "sanitizar_mimic_patients.ipynb", cells_mimic)

    # NHANES x4
    nhanes_specs = [
        (
            "sanitizar_nhanes_chronic.ipynb",
            "DATA_NHANES_CHRONIC",
            "NHANES — prevalencia de condiciones crónicas",
            "Tabla agregada por grupos (año, sexo, edad, raza): prevalencias (%) y errores.",
        ),
        (
            "sanitizar_nhanes_infectious.ipynb",
            "DATA_NHANES_INFECTIOUS",
            "NHANES — enfermedades infecciosas",
            "Datos de prevalencia/incidencia agregados; muchas columnas categóricas.",
        ),
        (
            "sanitizar_nhanes_dietary.ipynb",
            "DATA_NHANES_DIETARY",
            "NHANES — ingesta dietética media",
            "Estimaciones de ingesta; `Percent` o medidas similares como posible objetivo de regresión.",
        ),
        (
            "sanitizar_nhanes_oral.ipynb",
            "DATA_NHANES_ORAL",
            "NHANES — salud oral",
            "Prevalencias relacionadas con salud bucal.",
        ),
    ]
    nhanes_dir = next(ROOT.glob("3. NHANES*"))
    for fname, envk, title, desc in nhanes_specs:
        cells_n = [
            new_markdown_cell(
                f"# {title}\n\n## En qué consiste\n- {desc}\n"
                "- **Uso ML**: típicamente **regresión** si usas `Percent` como objetivo, o features para otros modelos.\n\n"
                "## Pasos\n1. Cargar CSV.\n2. Si existe `Percent`, será `y`.\n3. One-hot en categóricas; escalar numéricas.\n4. Mostrar antes/después y 10 filas."
            ),
            new_code_cell(SETUP),
            new_code_cell(
                f'''import pandas as pd
import numpy as np

path = data_path("{envk}")
df_original = pd.read_csv(path)
print("Forma:", df_original.shape)
print("Columnas:", list(df_original.columns))

target_col = "Percent" if "Percent" in df_original.columns else None
if target_col:
    y = pd.to_numeric(df_original[target_col], errors="coerce").values
    X = df_original.drop(columns=[target_col])
    print("[OBJETIVO] Columna Percent separada como y (regresion).")
else:
    X = df_original
    y = None
    print("[SIN OBJETIVO FIJO] Solo transformacion de features.")

# Sin eliminar columnas salvo vacias totalmente opcional — aqui conservamos todas las features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numericas:", len(num_cols), "| Categoricas:", len(cat_cols))

n_preview = min(12, X.shape[1])
print("\\n=== MUESTRA: 10 filas de X (primeras", n_preview, "columnas) ===")
print(X.iloc[:10, :n_preview].to_string())
'''
            ),
            MD_PIPELINE,
            new_code_cell(PIPELINE_FUN),
            new_code_cell(display_block()),
            MD_CIERRE,
        ]
        save(nhanes_dir / fname, cells_n)

    # Bike
    bike_dir = next(ROOT.glob("4. Bike*"))
    cells_bike = [
        new_markdown_cell(
            r"""# Bike Sharing — Regresión de demanda (conteo)

## En qué consiste
- Datos de alquiler de bicicletas; en este proyecto solo está **`sampleSubmission.csv`** (fechas + `count` de ejemplo).
- **Objetivo**: típicamente predecir **demanda horaria** (`count`) — regresión.

## Pasos
1. Derivar hora, día del mes, etc. de `datetime`.
2. Escalar numéricas; one-hot si hubiera categorías.
3. Mostrar antes/después y 10 filas."""
        ),
        new_code_cell(SETUP),
        new_code_cell(
            r'''import pandas as pd
import numpy as np

path = data_path("DATA_BIKE_SHARING")
df_original = pd.read_csv(path)
print("Forma:", df_original.shape)

if "datetime" in df_original.columns:
    df_original["datetime"] = pd.to_datetime(df_original["datetime"], errors="coerce")
    df_original["hour"] = df_original["datetime"].dt.hour
    df_original["dow"] = df_original["datetime"].dt.dayofweek
    df_original["month"] = df_original["datetime"].dt.month
    df_original = df_original.drop(columns=["datetime"])
    print("[TRANSFORMADA] datetime -> hour, dow, month")

y_col = "count" if "count" in df_original.columns else None
if y_col:
    y = pd.to_numeric(df_original[y_col], errors="coerce").values
    X = df_original.drop(columns=[y_col])
else:
    X = df_original
    y = np.zeros(len(X))

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numericas:", num_cols, "| Categoricas:", cat_cols)

n_preview = min(12, X.shape[1])
print("\n=== MUESTRA: 10 filas de X (primeras", n_preview, "columnas) ===")
print(X.iloc[:10, :n_preview].to_string())
'''
        ),
        MD_PIPELINE,
        new_code_cell(PIPELINE_FUN),
        new_code_cell(display_block()),
        MD_CIERRE,
    ]
    save(bike_dir / "sanitizar_bike_sharing.ipynb", cells_bike)

    # Adult
    col_names = """'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'"""
    adult_dir = next(ROOT.glob("5. Adult*"))
    cells_adult = [
        new_markdown_cell(
            r"""# Adult (Census Income) — Clasificación de ingreso

## En qué consiste
- **Censo USA** (1994): personas con atributos demográficos y laborales.
- **Objetivo**: predecir si **`income`** es `>50K` o `<=50K` (clasificación binaria).

## Pasos
1. Leer `adult.data` sin cabecera; `?` = faltante.
2. Codificar `income` a 0/1.
3. One-hot en categóricas; escalar numéricas.
4. Mostrar eliminados (ninguno salvo target), antes/después, 10 filas."""
        ),
        new_code_cell(SETUP),
        new_code_cell(
            f'''import pandas as pd
import numpy as np

col_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
path = data_path("DATA_ADULT_TRAIN")
df_original = pd.read_csv(path, names=col_names, na_values=" ?", skipinitialspace=True)
print("Forma:", df_original.shape)

# Objetivo: income binario
df_original["income"] = df_original["income"].astype(str).str.strip().str.rstrip(".")
df_original["income"] = df_original["income"].map({{"<=50K": 0, ">50K": 1}})
y = df_original["income"].values
X = df_original.drop(columns=["income"])
print("[ELIMINADA] columna income del X (es la etiqueta y).")

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numericas:", num_cols)
print("Categoricas (ejemplo):", cat_cols[:5], "... total", len(cat_cols))

n_preview = min(12, X.shape[1])
print("\\n=== MUESTRA: 10 filas de X (primeras", n_preview, "columnas) ===")
print(X.iloc[:10, :n_preview].to_string())
'''
        ),
        MD_PIPELINE,
        new_code_cell(PIPELINE_FUN),
        new_code_cell(display_block()),
        MD_CIERRE,
    ]
    save(adult_dir / "sanitizar_adult.ipynb", cells_adult)

    # Credit
    credit_dir = next(ROOT.glob("6. Credit*"))
    cells_credit = [
        new_markdown_cell(
            r"""# Credit Approval (UCI crx) — Clasificación aprobación

## En qué consiste
- Solicitudes de tarjeta con atributos **anonimizados** (A1–A15) y clase **A16** (+ / -).
- **Objetivo**: predecir aprobación (+) vs rechazo (-).

## Pasos
1. Leer `crx.data`; `?` = faltante.
2. `y` = A16 como 0/1.
3. Tipos mixtos → numéricas vs categóricas por dtype.
4. Antes/después y 10 filas."""
        ),
        new_code_cell(SETUP),
        new_code_cell(
            r'''import pandas as pd
import numpy as np

path = data_path("DATA_CREDIT_CRX")
df_original = pd.read_csv(path, header=None, na_values="?")
df_original.columns = [f"A{i}" for i in range(1, 16)] + ["target"]
print("Forma:", df_original.shape)

y = (df_original["target"] == "+").astype(int).values
X = df_original.drop(columns=["target"])
print("[ELIMINADA] target del feature matrix; y = 1 si credito aprobado (+).")

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numericas:", num_cols, "| Categoricas:", cat_cols)

n_preview = min(12, X.shape[1])
print("\n=== MUESTRA: 10 filas de X (primeras", n_preview, "columnas) ===")
print(X.iloc[:10, :n_preview].to_string())
'''
        ),
        MD_PIPELINE,
        new_code_cell(PIPELINE_FUN),
        new_code_cell(display_block()),
        MD_CIERRE,
    ]
    save(credit_dir / "sanitizar_credit_approval.ipynb", cells_credit)

    # Australian
    aus_dir = next(ROOT.glob("7. Statlog*"))
    cells_aus = [
        new_markdown_cell(
            r"""# Australian Credit (Statlog) — Clasificación

## En qué consiste
- **690** instancias; primera columna = **etiqueta** (0/1), resto atributos (mezcla).
- **Objetivo**: predecir la primera columna.

## Pasos
1. Leer separado por espacios.
2. y = col 0; X = resto.
3. Pipeline estándar; antes/después; 10 filas."""
        ),
        new_code_cell(SETUP),
        new_code_cell(
            r'''import pandas as pd
import numpy as np

path = data_path("DATA_AUSTRALIAN")
df_original = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
y = df_original.iloc[:, 0].astype(int).values
X = df_original.iloc[:, 1:].copy()
X.columns = [f"f{i}" for i in range(X.shape[1])]
print("[ELIMINADA] columna 0 del X porque es la etiqueta y.")
print("Forma X:", X.shape)

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numericas:", num_cols, "| Categoricas:", cat_cols)

n_preview = min(12, X.shape[1])
print("\n=== MUESTRA: 10 filas de X (primeras", n_preview, "columnas) ===")
print(X.iloc[:10, :n_preview].to_string())
'''
        ),
        MD_PIPELINE,
        new_code_cell(PIPELINE_FUN),
        new_code_cell(display_block()),
        MD_CIERRE,
    ]
    save(aus_dir / "sanitizar_australian_credit.ipynb", cells_aus)

    # Breast cancer
    bc_dir = next(ROOT.glob("8. Breast*"))
    cells_bc = [
        new_markdown_cell(
            r"""# Breast Cancer Wisconsin (Original) — Clasificación

## En qué consiste
- **699** muestras de tejido; **9** características (1–10) + **clase** (2 benigno, 4 maligno según UCI).
- **Objetivo**: diagnosticar maligno vs benigno.

## Pasos
1. Leer sin cabecera; `?` en bare nuclei = faltante.
2. Quitar **id** de muestra.
3. y = 1 si maligno (clase 4), 0 si benigno (2).
4. Pipeline; antes/después; 10 filas."""
        ),
        new_code_cell(SETUP),
        new_code_cell(
            r'''import pandas as pd
import numpy as np

cols = [
    "id", "clump", "cell_size", "cell_shape", "marginal", "epithelial",
    "bare_nuclei", "bland", "nucleoli", "mitoses", "class",
]
path = data_path("DATA_BREAST_CANCER")
df_original = pd.read_csv(path, header=None, names=cols, na_values="?")
print("Forma:", df_original.shape)

y = (df_original["class"] == 4).astype(int).values
X = df_original.drop(columns=["class"])
if "id" in X.columns:
    X = X.drop(columns=["id"])
    print("[ELIMINADA] id — identificador de muestra.")
print("[OBJETIVO] y=1 maligno (clase 4), y=0 benigno (clase 2).")

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Features numericas:", num_cols)

n_preview = min(12, X.shape[1])
print("\n=== MUESTRA: 10 filas de X (primeras", n_preview, "columnas) ===")
print(X.iloc[:10, :n_preview].to_string())
'''
        ),
        MD_PIPELINE,
        new_code_cell(PIPELINE_FUN),
        new_code_cell(display_block()),
        MD_CIERRE,
    ]
    save(bc_dir / "sanitizar_breast_cancer_wisconsin.ipynb", cells_bc)

    # Meningitis
    men_dir = next(ROOT.glob("9. Meningitis*"))
    cells_men = [
        new_markdown_cell(
            r"""# Meningitis (Kaggle) — Clasificación con faltantes

## En qué consiste
- Pacientes con variables clínicas y laboratorio; hay **valores faltantes**.
- **Objetivo ejemplo**: predecir **`Outcome`** (recuperación, etc.) o **`Diagnosis`**.

## Pasos
1. Quitar **Patient_ID**.
2. Codificar **Outcome** con códigos enteros.
3. Imputación + one-hot + escalado; antes/después; 10 filas."""
        ),
        new_code_cell(SETUP),
        new_code_cell(
            r'''import pandas as pd
import numpy as np

path = data_path("DATA_MENINGITIS")
df_original = pd.read_csv(path)
print("Forma:", df_original.shape)
print("Columnas:", list(df_original.columns))

if "Patient_ID" in df_original.columns:
    df_original = df_original.drop(columns=["Patient_ID"])
    print("[ELIMINADA] Patient_ID — identificador.")

if "Outcome" in df_original.columns:
    y = pd.Categorical(df_original["Outcome"]).codes
    X = df_original.drop(columns=["Outcome"])
    print("[OBJETIVO] Outcome codificado como enteros 0..k-1.")
else:
    X = df_original
    y = np.zeros(len(X))

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numericas:", num_cols)
print("Categoricas:", cat_cols)

n_preview = min(12, X.shape[1])
print("\n=== MUESTRA: 10 filas de X (primeras", n_preview, "columnas) ===")
print(X.iloc[:10, :n_preview].to_string())
'''
        ),
        MD_PIPELINE,
        new_code_cell(PIPELINE_FUN),
        new_code_cell(display_block()),
        MD_CIERRE,
    ]
    save(men_dir / "sanitizar_meningitis.ipynb", cells_men)

    print("Listo. Cuadernos generados en cada carpeta.")


if __name__ == "__main__":
    main()
