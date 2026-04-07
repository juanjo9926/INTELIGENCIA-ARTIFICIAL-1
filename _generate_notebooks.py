"""Genera notebooks de sanitizacion (ejecutar una vez: python _generate_notebooks.py)."""
from __future__ import annotations

import shutil
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent

COMMON_SETUP = r'''import sys
from pathlib import Path

# Raiz del proyecto (donde esta .env y load_project_env.py)
def _project_root() -> Path:
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
print("PROJECT_ROOT:", ROOT)
'''

PIPELINE_CLASS = r'''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X, numeric, categorical):
    """Imputacion + escalado numerico; imputacion + one-hot categoricos."""
    num = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num, numeric),
            ("cat", cat, categorical),
        ],
        remainder="drop",
    )
'''

PIPELINE_REG = r'''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor_regression(X, numeric, categorical):
    num = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop",
    )
'''


def save_nb(path: Path, cells: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook()
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata["kernelspec"].update(
        display_name="Python 3",
        language="python",
        name="python3",
    )
    nb.cells = cells
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def main() -> None:
    notebooks: list[tuple[Path, list]] = []

    # --- Ames (regresion) ---
    notebooks.append(
        (
            ROOT / "1. Ames Housing Dataset" / "sanitizar_ames_housing.ipynb",
            [
                new_markdown_cell(
                    "# Ames Housing — sanitizado para regresion / redes\n\n"
                    "Objetivo: `SalePrice` (regresion). Variables categoricas: one-hot; numericas: imputacion + estandarizacion."
                ),
                new_code_cell(COMMON_SETUP),
                new_code_cell(
                    'import os\nimport pandas as pd\nimport numpy as np\n\n'
                    'csv_path = data_path("DATA_AMES_TRAIN")\n'
                    "df = pd.read_csv(csv_path)\n"
                    'target_col = "SalePrice"\n'
                    "y = df[target_col].astype(float).values\n"
                    "X = df.drop(columns=[target_col])\n"
                    "# Quitar ID si existe\n"
                    "for c in ('Id', 'Order', 'PID'):\n"
                    "    if c in X.columns:\n"
                    "        X = X.drop(columns=[c])\n"
                    "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
                    "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                    "print(X.shape, len(num_cols), len(cat_cols))"
                ),
                new_code_cell(PIPELINE_REG),
                new_code_cell(
                    "prep = build_preprocessor_regression(X, num_cols, cat_cols)\n"
                    "X_matrix = prep.fit_transform(X)\n"
                    "print('Matriz lista:', X_matrix.shape, '| y:', y.shape)\n"
                    "# Listo para train_test_split, MLPRegressor, etc."
                ),
            ],
        )
    )

    # --- MIMIC PATIENTS ---
    mimic_dir = next(ROOT.glob("2. MIMIC*"))
    notebooks.append(
        (
            mimic_dir / "sanitizar_mimic_patients.ipynb",
            [
                new_markdown_cell(
                    "# MIMIC-III PATIENTS — clasificacion (expire_flag)\n\n"
                    "Fechas codificadas como ordinal; genero one-hot; objetivo: `expire_flag`."
                ),
                new_code_cell(COMMON_SETUP),
                new_code_cell(
                    'import pandas as pd\nimport numpy as np\n\n'
                    'path = data_path("DATA_MIMIC_PATIENTS")\n'
                    "df = pd.read_csv(path)\n"
                    "df = df.drop(columns=[c for c in ('row_id', 'subject_id') if c in df.columns])\n"
                    "for col in ('dob', 'dod', 'dod_hosp', 'dod_ssn'):\n"
                    "    if col in df.columns:\n"
                    "        df[col] = pd.to_datetime(df[col], errors='coerce')\n"
                    "        df[col + '_ordinal'] = df[col].apply(lambda t: t.toordinal() if pd.notna(t) else np.nan)\n"
                    "        df = df.drop(columns=[col])\n"
                    'y = df["expire_flag"].astype(int).values if "expire_flag" in df.columns else None\n'
                    'X = df.drop(columns=["expire_flag"]) if y is not None else df\n'
                    "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
                    "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                    "print(X.shape, y.shape if y is not None else 'sin y')"
                ),
                new_code_cell(PIPELINE_CLASS),
                new_code_cell(
                    "if y is not None:\n"
                    "    prep = build_preprocessor(X, num_cols, cat_cols)\n"
                    "    X_matrix = prep.fit_transform(X)\n"
                    "    print('Matriz:', X_matrix.shape)\n"
                    "else:\n"
                    "    print('Sin columna objetivo')"
                ),
            ],
        )
    )

    nhanes_specs = [
        (
            "sanitizar_nhanes_chronic.ipynb",
            "DATA_NHANES_CHRONIC",
            "NHANES — condiciones cronicas (prevalencia)",
        ),
        (
            "sanitizar_nhanes_infectious.ipynb",
            "DATA_NHANES_INFECTIOUS",
            "NHANES — enfermedades infecciosas",
        ),
        (
            "sanitizar_nhanes_dietary.ipynb",
            "DATA_NHANES_DIETARY",
            "NHANES — ingesta dietetica media",
        ),
        (
            "sanitizar_nhanes_oral.ipynb",
            "DATA_NHANES_ORAL",
            "NHANES — salud oral",
        ),
    ]
    nhanes_dir = next(ROOT.glob("3. NHANES*"))
    for fname, envk, title in nhanes_specs:
        notebooks.append(
            (
                nhanes_dir / fname,
                [
                    new_markdown_cell(
                        f"# {title}\n\n"
                        "Tabla agregada: one-hot en columnas categoricas; `Percent` como posible objetivo de regresion."
                    ),
                    new_code_cell(COMMON_SETUP),
                    new_code_cell(
                        'import pandas as pd\nimport numpy as np\n\n'
                        f'path = data_path("{envk}")\n'
                        "df = pd.read_csv(path)\n"
                        "# Objetivo opcional: Percent numerico\n"
                        'target_col = "Percent" if "Percent" in df.columns else None\n'
                        "if target_col:\n"
                        "    y = pd.to_numeric(df[target_col], errors='coerce').values\n"
                        "    X = df.drop(columns=[target_col])\n"
                        "else:\n"
                        "    X, y = df, None\n"
                        "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
                        "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                        "print(X.shape)"
                    ),
                    new_code_cell(PIPELINE_REG),
                    new_code_cell(
                        "prep = build_preprocessor_regression(X, num_cols, cat_cols)\n"
                        "X_matrix = prep.fit_transform(X)\n"
                        "print('X procesada:', X_matrix.shape, '| y:', None if y is None else y.shape)"
                    ),
                ],
            )
        )

    # --- Bike (solo sample submission en repo) ---
    bike_dir = next(ROOT.glob("4. Bike*"))
    notebooks.append(
        (
            bike_dir / "sanitizar_bike_sharing.ipynb",
            [
                new_markdown_cell(
                    "# Bike sharing — regresion sobre `count`\n\n"
                    "En este repo solo esta `sampleSubmission.csv`. Si anades `hour.csv` o `train.csv`, "
                    "ajusta `DATA_BIKE_SHARING` en `.env`."
                ),
                new_code_cell(COMMON_SETUP),
                new_code_cell(
                    'import pandas as pd\nimport numpy as np\n\n'
                    'path = data_path("DATA_BIKE_SHARING")\n'
                    "df = pd.read_csv(path)\n"
                    'if "datetime" in df.columns:\n'
                    "    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')\n"
                    "    df['hour'] = df['datetime'].dt.hour\n"
                    "    df['dow'] = df['datetime'].dt.dayofweek\n"
                    "    df['month'] = df['datetime'].dt.month\n"
                    "    df = df.drop(columns=['datetime'])\n"
                    'y_col = "count" if "count" in df.columns else None\n'
                    "if y_col:\n"
                    "    y = pd.to_numeric(df[y_col], errors='coerce').values\n"
                    "    X = df.drop(columns=[y_col])\n"
                    "else:\n"
                    "    X, y = df, None\n"
                    "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
                    "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                    "print(X.shape)"
                ),
                new_code_cell(PIPELINE_REG),
                new_code_cell(
                    "prep = build_preprocessor_regression(X, num_cols, cat_cols)\n"
                    "X_matrix = prep.fit_transform(X)\n"
                    "print('Shape:', X_matrix.shape)"
                ),
            ],
        )
    )

    # --- Adult ---
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    cols_repr = repr(col_names)
    adult_dir = next(ROOT.glob("5. Adult*"))
    notebooks.append(
        (
            adult_dir / "sanitizar_adult.ipynb",
            [
                new_markdown_cell(
                    "# Adult / Census Income — clasificacion binaria (>50K)\n\n"
                    "Valores `?` tratados como NaN; one-hot en categoricos; escalado en numericos."
                ),
                new_code_cell(COMMON_SETUP),
                new_code_cell(
                    f"import pandas as pd\nimport numpy as np\n\n"
                    f"col_names = {cols_repr}\n"
                    'path = data_path("DATA_ADULT_TRAIN")\n'
                    "df = pd.read_csv(path, names=col_names, na_values=' ?', skipinitialspace=True)\n"
                    "df['income'] = df['income'].astype(str).str.strip().str.rstrip('.')\n"
                    "df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})\n"
                    "y = df['income'].values\n"
                    "X = df.drop(columns=['income'])\n"
                    "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
                    "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                    "print(X.shape, y.shape)"
                ),
                new_code_cell(PIPELINE_CLASS),
                new_code_cell(
                    "prep = build_preprocessor(X, num_cols, cat_cols)\n"
                    "X_matrix = prep.fit_transform(X)\n"
                    "print('Listo para clasificacion:', X_matrix.shape)"
                ),
            ],
        )
    )

    # --- Credit crx ---
    credit_dir = next(ROOT.glob("6. Credit*"))
    notebooks.append(
        (
            credit_dir / "sanitizar_credit_approval.ipynb",
            [
                new_markdown_cell(
                    "# Credit Approval (crx) — clasificacion A16 (+/-)\n\n"
                    "Mezcla de continuos y categoricos; `?` como faltante."
                ),
                new_code_cell(COMMON_SETUP),
                new_code_cell(
                    'import pandas as pd\nimport numpy as np\n\n'
                    'path = data_path("DATA_CREDIT_CRX")\n'
                    "df = pd.read_csv(path, header=None, na_values='?')\n"
                    "df.columns = [f'A{i}' for i in range(1, 16)] + ['target']\n"
                    "df['target'] = (df['target'] == '+').astype(int)\n"
                    "y = df['target'].values\n"
                    "X = df.drop(columns=['target'])\n"
                    "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
                    "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                    "print(X.shape)"
                ),
                new_code_cell(PIPELINE_CLASS),
                new_code_cell(
                    "prep = build_preprocessor(X, num_cols, cat_cols)\n"
                    "X_matrix = prep.fit_transform(X)\n"
                    "print(X_matrix.shape)"
                ),
            ],
        )
    )

    # --- Australian ---
    aus_dir = next(ROOT.glob("7. Statlog*"))
    notebooks.append(
        (
            aus_dir / "sanitizar_australian_credit.ipynb",
            [
                new_markdown_cell(
                    "# Australian Credit (Statlog) — clasificacion\n\n"
                    "Primera columna etiqueta; resto mezcla numerica/categorica inferida por tipo."
                ),
                new_code_cell(COMMON_SETUP),
                new_code_cell(
                    'import pandas as pd\nimport numpy as np\n\n'
                    'path = data_path("DATA_AUSTRALIAN")\n'
                    "df = pd.read_csv(path, sep=r'\\s+', header=None)\n"
                    "y = df.iloc[:, 0].astype(int).values\n"
                    "X = df.iloc[:, 1:]\n"
                    "X.columns = [f'f{i}' for i in range(X.shape[1])]\n"
                    "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
                    "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                    "print(X.shape, y.shape)"
                ),
                new_code_cell(PIPELINE_CLASS),
                new_code_cell(
                    "prep = build_preprocessor(X, num_cols, cat_cols)\n"
                    "X_matrix = prep.fit_transform(X)\n"
                    "print(X_matrix.shape)"
                ),
            ],
        )
    )

    # --- Breast cancer ---
    bc_cols = [
        "id",
        "clump",
        "cell_size",
        "cell_shape",
        "marginal",
        "epithelial",
        "bare_nuclei",
        "bland",
        "nucleoli",
        "mitoses",
        "class",
    ]
    bc_dir = next(ROOT.glob("8. Breast*"))
    notebooks.append(
        (
            bc_dir / "sanitizar_breast_cancer_wisconsin.ipynb",
            [
                new_markdown_cell(
                    "# Breast Cancer Wisconsin — clasificacion (benigno/maligno)\n\n"
                    "Clase 2/4 codificada a 0/1; `bare_nuclei` puede tener `?`."
                ),
                new_code_cell(COMMON_SETUP),
                new_code_cell(
                    f"import pandas as pd\nimport numpy as np\n\n"
                    f"cols = {bc_cols!r}\n"
                    'path = data_path("DATA_BREAST_CANCER")\n'
                    "df = pd.read_csv(path, header=None, names=cols, na_values='?')\n"
                    "df = df.drop(columns=['id'])\n"
                    "y = (df['class'] == 4).astype(int).values\n"
                    "X = df.drop(columns=['class'])\n"
                    "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
                    "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                    "print(X.shape, y.shape)"
                ),
                new_code_cell(PIPELINE_CLASS),
                new_code_cell(
                    "prep = build_preprocessor(X, num_cols, cat_cols)\n"
                    "X_matrix = prep.fit_transform(X)\n"
                    "print('Features listas:', X_matrix.shape)"
                ),
            ],
        )
    )

    # --- Meningitis ---
    men_dir = next(ROOT.glob("9. Meningitis*"))
    notebooks.append(
        (
            men_dir / "sanitizar_meningitis.ipynb",
            [
                new_markdown_cell(
                    "# Meningitis — clasificacion (p. ej. `Outcome` o `Diagnosis`)\n\n"
                    "Imputacion de faltantes; one-hot en categoricos; escalado en numericos."
                ),
                new_code_cell(COMMON_SETUP),
                new_code_cell(
                    'import pandas as pd\nimport numpy as np\n\n'
                    'path = data_path("DATA_MENINGITIS")\n'
                    "df = pd.read_csv(path)\n"
                    "# Objetivo: Outcome codificado\n"
                    'if "Outcome" in df.columns:\n'
                    "    y = pd.Categorical(df['Outcome']).codes\n"
                    "    X = df.drop(columns=['Outcome'])\n"
                    "else:\n"
                    "    X, y = df, None\n"
                    'if "Patient_ID" in X.columns:\n'
                    "    X = X.drop(columns=['Patient_ID'])\n"
                    "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
                    "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                    "print(X.shape, y.shape if y is not None else None)"
                ),
                new_code_cell(PIPELINE_CLASS),
                new_code_cell(
                    "if y is not None:\n"
                    "    prep = build_preprocessor(X, num_cols, cat_cols)\n"
                    "    X_matrix = prep.fit_transform(X)\n"
                    "    print(X_matrix.shape)\n"
                    "else:\n"
                    "    print('Definir columna objetivo')"
                ),
            ],
        )
    )

    for path, cells in notebooks:
        save_nb(path, cells)
        print("Wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
