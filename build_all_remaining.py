# -*- coding: utf-8 -*-
"""Genera notebooks 3-9 (NHANES, Bike, Adult, Credit, Australian, Breast, Meningitis)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebook_helpers import project_root
from datasets_2_9_notebooks import ROOT_STR, code, md, save_nb

ROOT = project_root()


def nb_header(titulo: str, desc: str, tarea: str, objetivo: str) -> list:
    """Markdown seccion 0 generico."""
    return [
        md(
            f"""# {titulo}

## SECCIÓN 0 — DESCRIPCIÓN DEL DATASET

{desc}

### Tarea de machine learning
- **Tipo de tarea**: {tarea}
- **Variable objetivo**: {objetivo}

Las siguientes celdas cargan datos y muestran `head`, `shape`, `dtypes` y `describe`."""
        )
    ]


def std_imports() -> str:
    return f"""import warnings  # Avisos
warnings.filterwarnings("ignore")  # Ocultar
from pathlib import Path  # Rutas
import numpy as np  # Numerico
import pandas as pd  # Tablas
import matplotlib.pyplot as plt  # Graficos
import seaborn as sns  # Stats
from IPython.display import display  # Jupyter
from sklearn.model_selection import train_test_split  # Split
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Escalado
from sklearn.linear_model import LinearRegression, LogisticRegression  # Modelos
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score  # Metricas
ROOT_NB = Path(r"{ROOT_STR}")  # Raiz proyecto
"""


def seccion1_template() -> str:
    return """df1 = df.copy()  # Copia para limpiar
print("Nulos ANTES por columna (top):\\n", df1.isnull().sum().sort_values(ascending=False).head(15))  # Conteo
df1 = df1.drop_duplicates()  # Eliminar filas duplicadas
nr = df1.isnull().mean()  # Fraccion nulos
drop_null = nr[nr > 0.40].index.tolist()  # Columnas >40% nulos
print("Eliminadas por >40% nulos:", drop_null)  # Motivo: poca informacion
df1 = df1.drop(columns=drop_null, errors="ignore")  # Borrar
num_cols = df1.select_dtypes(include=[np.number]).columns.tolist()  # Numericas
cat_cols = [c for c in df1.columns if c not in num_cols]  # Categoricas
for c in num_cols:
    df1[c] = df1[c].fillna(df1[c].median())  # Mediana (robusta a outliers)
for c in cat_cols:
    if df1[c].isnull().any():
        mod = df1[c].mode()
        df1[c] = df1[c].fillna(mod.iloc[0] if len(mod) else "")  # Moda
print("Nulos DESPUES (total):", int(df1.isnull().sum().sum()))  # Debe ser 0
assert df1.isnull().sum().sum() == 0
df_clean = df1.copy()
"""


def seccion2_clf(yname: str) -> str:
    return f"""y = df_clean["{yname}"]  # Objetivo
X = df_clean.drop(columns=["{yname}"])  # Features
print("X.shape", X.shape, "y.shape", y.shape)  # Dimensiones
print("Distribucion de clases:\\n", y.value_counts())  # Conteos
plt.figure(figsize=(6,5))  # Figura
y.astype(str).value_counts().plot(kind="bar")  # Barras
plt.title("Distribucion de clases")  # Titulo
plt.tight_layout()  # Ajuste
plt.show()  # Mostrar
num_x = X.select_dtypes(include=[np.number]).columns  # Numericas
if len(num_x) > 0 and y.dtype != object:
    le = LabelEncoder()  # Codificar y para correlacion
    y_num = le.fit_transform(y.astype(str))  # y numerico
    corr = X[num_x].corrwith(pd.Series(y_num, index=X.index)).abs().sort_values(ascending=False)  # Correlaciones
    print("Top 5 correlaciones (aprox) con y:", corr.head(5))  # Top 5
    sns.heatmap(pd.concat([X[num_x[:min(5,len(num_x))]], pd.Series(y_num, name="y")], axis=1).corr(), annot=True)  # Mapa
    plt.show()
elif len(num_x) > 0:
    corr = X[num_x].apply(lambda s: pd.factorize(y)[0])  # Fallback
"""


def seccion2_reg(yname: str) -> str:
    return f"""y = pd.to_numeric(df_clean["{yname}"], errors="coerce")  # Objetivo numerico
X = df_clean.drop(columns=["{yname}"])  # Features
print("X.shape", X.shape, "y.shape", y.shape)  # Shapes
num_x = X.select_dtypes(include=[np.number]).columns  # Numericas
if len(num_x) > 0:
    corr = X[num_x].corrwith(y).abs().sort_values(ascending=False)  # Corr con precio
    print("Top 5 correlaciones con y:", corr.head(5))  # Top 5
plt.figure(figsize=(5,3))  # Figura
plt.hist(y.dropna(), bins=30)  # Histograma
plt.title("Distribucion de y (regresion)")  # Titulo
plt.show()  # Mostrar
"""


def seccion3_enc() -> str:
    return """X_enc = X.copy()  # Copia
non_num = [c for c in X_enc.columns if not pd.api.types.is_numeric_dtype(X_enc[c])]  # Categoricas
print("Categoricas (antes encoding):", non_num)  # Lista
X_enc = pd.get_dummies(X_enc, columns=non_num, drop_first=False)  # One-hot (nominal)
print("Columnas despues encoding:", X_enc.shape[1])  # Total
display(X_enc.head(5))  # Primeras filas
"""


def seccion4_scaler() -> str:
    return """scaler = StandardScaler()  # Escalador
X_scaled = scaler.fit_transform(X_enc)  # Ajuste global
X_scaled_df = pd.DataFrame(X_scaled, columns=X_enc.columns, index=X_enc.index)  # DF
antes = X_enc.iloc[:5, :min(6, X_enc.shape[1])]  # 5 filas antes
despues = X_scaled_df.iloc[:5, :min(6, X_scaled_df.shape[1])]  # 5 filas despues
display(pd.concat([antes, despues], axis=1, keys=["ANTES","DESPUES"]))  # Comparativa
"""


def seccion5_split(clf: bool) -> str:
    if clf:
        return """X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)  # Estratificado
print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # Tamano
"""
    return """X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)  # Regresion
print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # Tamano
"""


def build_nhanes() -> list:
    """NHANES: CSV agregado; multiclass Measure (3 enfermedades)."""
    folder = list(ROOT.glob("3. NHANES*"))[0]
    csvp = folder / "NHANES_Select_Chronic_Conditions_Prevalence_Estimates.csv"
    cells = nb_header(
        "NHANES — condiciones cronicas (tabla agregada)",
        "Datos **CDC/NHANES** en formato **CSV agregado** (no microdatos individuales). No hay archivo `.XPT` en esta carpeta; se usa `pd.read_csv`. "
        "Las filas son grupos poblacionales (año, sexo, edad, raza) y medidas de prevalencia.",
        "clasificación multiclase (predecir `Measure`: Obesity, Hypertension, etc.)",
        "`Measure` — tipo de condición crónica reportada.",
    )
    cells.append(
        code(
            std_imports()
            + f"""
df = pd.read_csv(r"{str(csvp).replace(chr(92), '/')}")  # Cargar CSV
print("head:"); display(df.head(10))  # Muestra
print("shape:", df.shape)  # Forma
print("dtypes:\\n", df.dtypes)  # Tipos
display(df.describe(include="all").T)  # Estadisticos
"""
        )
    )
    cells.append(md("## SECCIÓN 1 — SANEAMIENTO\n\nMediana/modas; eliminar >40% nulos; sin duplicados exactos."))
    cells.append(code(seccion1_template()))
    cells.append(md("## SECCIÓN 2 — X e y\n\n**y** = `Measure` (3 clases). **X** = resto."))
    cells.append(
        code(
            seccion2_clf("Measure").replace('y = df_clean["Measure"]', 'y = df_clean["Measure"]  # Tres clases')
        )
    )
    cells.append(md("## SECCIÓN 3 — ENCODING\n\nOne-hot del resto de categorías; `Measure` ya separado como y."))
    cells.append(code(seccion3_enc()))
    cells.append(md("## SECCIÓN 4 — NORMALIZACIÓN"))
    cells.append(code(seccion4_scaler()))
    cells.append(md("## SECCIÓN 5 — SPLIT"))
    cells.append(code(seccion5_split(True)))
    cells.append(md("## SECCIÓN 6 — LogisticRegression"))
    cells.append(
        code(
            """y_enc = LabelEncoder().fit_transform(y.astype(str))  # Etiquetas 0..K-1
clf = LogisticRegression(max_iter=1000)  # Clasificador
clf.fit(X_train, y_train)  # Entrenar (y string -> sklearn acepta)
acc = accuracy_score(y_test, clf.predict(X_test))  # Precision
print("Accuracy:", acc)  # Resultado
score_nhanes = acc  # Score
"""
        )
    )
    cells.append(md("## SECCIÓN 7 — GUARDADO"))
    cells.append(
        code(
            f"""out = Path(r"{str(folder).replace(chr(92), '/')}")  # Carpeta
pd.concat([X_scaled_df, y.rename("Measure")], axis=1).to_csv(out / "nhanes_clean.csv", index=False)  # Limpio
X_train.to_csv(out / "X_train.csv", index=False)  # Train
X_test.to_csv(out / "X_test.csv", index=False)  # Test
y_train.to_csv(out / "y_train.csv", index=True, header=True)  # y
y_test.to_csv(out / "y_test.csv", index=True, header=True)  # y test
print("Archivos guardados en", out)  # Ok
"""
        )
    )
    cells.append(md("## SECCIÓN 8 — RESUMEN"))
    cells.append(
        code(
            """display(df_clean.head(10))  # Muestra
assert df_clean.isnull().sum().sum() == 0  # Sin nulos
assert np.all(np.isfinite(X_scaled_df.values))  # Finitos
print("RESUMEN NHANES | Accuracy:", round(acc, 4))  # Resumen
"""
        )
    )
    return cells


def build_bike() -> list:
    folder = list(ROOT.glob("4. Bike*"))[0]
    csvp = folder / "sampleSubmission.csv"
    cells = nb_header(
        "Bike Sharing (Washington DC)",
        "**Nota:** Solo esta disponible `sampleSubmission.csv` en la carpeta (plantilla Kaggle con `count` en 0). "
        "Para un modelo real de regresión de `cnt`/`count`, descargue `hour.csv` o `train.csv` del UCI/Kaggle y actualice la ruta. "
        "Este flujo demuestra el pipeline.",
        "regresión (target `count`)",
        "`count` — alquileres de bicicletas (en la plantilla muchos ceros).",
    )
    cells.append(
        code(
            std_imports()
            + f"""
df = pd.read_csv(r"{str(csvp).replace(chr(92), '/')}")  # Cargar
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")  # Fecha
df["hour"] = df["datetime"].dt.hour  # Hora
df["dow"] = df["datetime"].dt.dayofweek  # Dia semana
df["month"] = df["datetime"].dt.month  # Mes
df = df.drop(columns=["datetime"])  # Quitar fecha cruda
display(df.head(10))  # Muestra
print(df.shape, df.dtypes)  # Info
display(df.describe(include="all").T)  # Stats
"""
        )
    )
    cells.append(md("## SECCIÓN 1 — SANEAMIENTO"))
    cells.append(code(seccion1_template()))
    cells.append(md("## SECCIÓN 2 — X e y"))
    cells.append(code(seccion2_reg("count")))
    cells.append(md("## SECCIÓN 3 — ENCODING"))
    cells.append(code(seccion3_enc()))
    cells.append(md("## SECCIÓN 4"))
    cells.append(code(seccion4_scaler()))
    cells.append(md("## SECCIÓN 5"))
    cells.append(code(seccion5_split(False)))
    cells.append(md("## SECCIÓN 6 — LinearRegression"))
    cells.append(
        code(
            """reg = LinearRegression()  # Regresion
reg.fit(X_train, y_train)  # Entrenar
pred = reg.predict(X_test)  # Predecir
print("MSE:", mean_squared_error(y_test, pred), "R2:", r2_score(y_test, pred))  # Metricas
score_bike = r2_score(y_test, pred)  # Score
"""
        )
    )
    cells.append(md("## SECCIÓN 7"))
    cells.append(
        code(
            f"""out = Path(r"{str(folder).replace(chr(92), '/')}")  # Carpeta
pd.concat([X_scaled_df, y.rename("count")], axis=1).to_csv(out / "bike_sharing_clean.csv", index=False)  # Limpio
X_train.to_csv(out / "X_train.csv", index=False)  # Train
X_test.to_csv(out / "X_test.csv", index=False)  # Test
y_train.to_csv(out / "y_train.csv", index=True, header=True)  # y
y_test.to_csv(out / "y_test.csv", index=True, header=True)  # y test
print("Guardado bike_sharing_clean.csv")  # Ok
"""
        )
    )
    cells.append(md("## SECCIÓN 8"))
    cells.append(
        code(
            """display(df_clean.head(10))  # Muestra
assert df_clean.isnull().sum().sum() == 0
print("RESUMEN Bike | R2:", score_bike)  # Resumen
"""
        )
    )
    return cells


def build_adult() -> list:
    """Adult Census Income."""
    folder = list(ROOT.glob("5. Adult*"))[0]
    csvp = folder / "adult" / "adult.data"
    cells = nb_header(
        "Adult Census Income",
        "Extracto del **censo USA 1994** (UCI). Atributos demográficos y laborales.",
        "clasificación binaria (ingreso >50K vs <=50K)",
        "`income` codificado como etiquetas de texto.",
    )
    cols = """age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income""".split(",")
    cells.append(
        code(
            std_imports()
            + f"""
col_names = {cols!r}  # Nombres columnas
df = pd.read_csv(r"{str(csvp).replace(chr(92), '/')}", names=col_names, na_values=" ?", skipinitialspace=True)  # Cargar
display(df.head(10))  # Muestra
print(df.shape, "\\n", df.dtypes)  # Forma y tipos
display(df.describe(include="all").T)  # Describe
"""
        )
    )
    cells.append(md("## SECCIÓN 1 — SANEAMIENTO"))
    cells.append(code(seccion1_template()))
    cells.append(md("## SECCIÓN 2 — X e y\n\n`y` = `income` mapeado a 0/1."))
    cells.append(
        code(
            """df_clean["income_bin"] = df_clean["income"].astype(str).str.strip().str.rstrip(".").map({"<=50K": 0, ">50K": 1})  # Binario
y = df_clean["income_bin"]  # Vector objetivo 0/1
X = df_clean.drop(columns=["income", "income_bin"])  # Sin columna objetivo ni texto
print("X.shape", X.shape, "y.shape", y.shape)  # Shapes
print(y.value_counts())  # Clases
plt.figure(figsize=(4,3))  # Figura
y.value_counts().plot(kind="bar")  # Barras
plt.title("Clases income 0/1")  # Titulo
plt.show()  # Grafico
num_x = X.select_dtypes(include=[np.number]).columns  # Numericas
corr = X[num_x].corrwith(y).abs().sort_values(ascending=False)  # Correlacion
print("Top 5 correlaciones:", corr.head(5))  # Top 5
sns.heatmap(pd.concat([X[num_x[:5]], y.rename("y")], axis=1).corr(), annot=True)  # Heatmap
plt.show()
"""
        )
    )
    cells.append(md("## SECCIÓN 3 — ENCODING"))
    cells.append(code(seccion3_enc()))
    cells.append(md("## SECCIÓN 4"))
    cells.append(code(seccion4_scaler()))
    cells.append(md("## SECCIÓN 5"))
    cells.append(code(seccion5_split(True)))
    cells.append(md("## SECCIÓN 6 — LogisticRegression"))
    cells.append(
        code(
            """clf = LogisticRegression(max_iter=1000)  # Modelo
clf.fit(X_train, y_train)  # Entrenar
acc = accuracy_score(y_test, clf.predict(X_test))  # Accuracy
print("Accuracy test:", acc)  # Resultado
score_adult = acc  # Score
"""
        )
    )
    cells.append(md("## SECCIÓN 7"))
    cells.append(
        code(
            f"""out = Path(r"{str(list(ROOT.glob('5. Adult*'))[0]).replace(chr(92), '/')}")  # Carpeta
pd.concat([X_scaled_df, y.rename("income_bin")], axis=1).to_csv(out / "adult_census_clean.csv", index=False)  # Limpio
X_train.to_csv(out / "X_train.csv", index=False)  # Train
X_test.to_csv(out / "X_test.csv", index=False)  # Test
y_train.to_csv(out / "y_train.csv", index=True, header=True)  # y
y_test.to_csv(out / "y_test.csv", index=True, header=True)  # y test
print("Guardado adult_census_clean.csv")  # Ok
"""
        )
    )
    cells.append(md("## SECCIÓN 8"))
    cells.append(
        code(
            """display(df_clean.head(10))  # Muestra
assert df_clean.isnull().sum().sum() == 0
print("RESUMEN Adult | Accuracy:", round(acc, 4))  # Resumen
"""
        )
    )
    return cells


def build_credit() -> list:
    folder = list(ROOT.glob("6. Credit*"))[0]
    csvp = folder / "credit+approval" / "crx.data"
    cells = nb_header(
        "Credit Approval (UCI crx)",
        "Solicitudes de tarjeta de crédito; **atributos anonimizados** (A1–A15) por confidencialidad.",
        "clasificación binaria",
        "Última columna: `+` aprobado, `-` rechazado.",
    )
    cells.append(
        code(
            std_imports()
            + f"""
df = pd.read_csv(r"{str(csvp).replace(chr(92), '/')}", header=None, na_values="?")  # Sin cabecera
df.columns = [f"A{{i}}" for i in range(1, 16)] + ["target"]  # Nombres A1..A15 + target
display(df.head(10))  # Muestra
print(df.shape, df.dtypes)  # Info
display(df.describe(include="all").T)  # Stats
"""
        )
    )
    cells.append(md("## SECCIÓN 1"))
    cells.append(code(seccion1_template()))
    cells.append(md("## SECCIÓN 2\n\ny = `target` (+/-)."))
    cells.append(
        code(
            """y = (df_clean["target"] == "+").astype(int)  # 1 aprobado
X = df_clean.drop(columns=["target"])  # Features
print("X.shape", X.shape, "y.shape", y.shape)  # Shapes
print(y.value_counts())  # Distribucion
plt.figure(figsize=(4,3))  # Figura
y.value_counts().plot(kind="bar")  # Barras
plt.show()  # Grafico
num_x = X.select_dtypes(include=[np.number]).columns  # Numericas
if len(num_x) > 0:
    corr = X[num_x].corrwith(y).abs().sort_values(ascending=False)  # Corr
    print("Top 5:", corr.head(5))  # Top 5
"""
        )
    )
    cells.append(md("## SECCIÓN 3"))
    cells.append(code(seccion3_enc()))
    cells.append(md("## SECCIÓN 4"))
    cells.append(code(seccion4_scaler()))
    cells.append(md("## SECCIÓN 5"))
    cells.append(code(seccion5_split(True)))
    cells.append(md("## SECCIÓN 6"))
    cells.append(
        code(
            """clf = LogisticRegression(max_iter=1000)  # Modelo
clf.fit(X_train, y_train)  # Entrenar
acc = accuracy_score(y_test, clf.predict(X_test))  # Accuracy
print("Accuracy:", acc)  # Resultado
score_credit = acc  # Score
"""
        )
    )
    cells.append(md("## SECCIÓN 7"))
    cells.append(
        code(
            f"""out = Path(r"{str(list(ROOT.glob('6. Credit*'))[0]).replace(chr(92), '/')}")  # Carpeta
pd.concat([X_scaled_df, y.rename("approved")], axis=1).to_csv(out / "credit_approval_clean.csv", index=False)  # Limpio
X_train.to_csv(out / "X_train.csv", index=False)  # Train
X_test.to_csv(out / "X_test.csv", index=False)  # Test
y_train.to_csv(out / "y_train.csv", index=True, header=True)  # y
y_test.to_csv(out / "y_test.csv", index=True, header=True)  # y test
print("Guardado credit_approval_clean.csv")  # Ok
"""
        )
    )
    cells.append(md("## SECCIÓN 8"))
    cells.append(
        code(
            """display(df_clean.head(10))  # Muestra
assert df_clean.isnull().sum().sum() == 0
print("RESUMEN Credit | Acc:", round(acc, 4))  # Resumen
"""
        )
    )
    return cells


def build_australian() -> list:
    folder = list(ROOT.glob("7. Statlog*"))[0]
    csvp = folder / "statlog+australian+credit+approval" / "australian.dat"
    cells = nb_header(
        "Statlog Australian Credit",
        "Credit scoring en Australia (UCI Statlog); similar a Credit Approval.",
        "clasificación binaria",
        "Primera columna = etiqueta (0/1), última en algunas descripciones — aquí columna 0 es la etiqueta.",
    )
    cells.append(
        code(
            std_imports()
            + f"""
df = pd.read_csv(r"{str(csvp).replace(chr(92), '/')}", sep=r"\\s+", header=None, engine="python")  # Regex: uno o mas espacios
display(df.head(10))  # Muestra
print(df.shape)  # Forma
print(df.dtypes)  # Tipos
display(df.describe(include="all").T)  # Stats
"""
        )
    )
    cells.append(md("## SECCIÓN 1"))
    cells.append(code(seccion1_template()))
    cells.append(md("## SECCIÓN 2 — Primera columna = y"))
    cells.append(
        code(
            """y = df_clean.iloc[:, 0].astype(int)  # Primera columna etiqueta
X = df_clean.iloc[:, 1:].copy()  # Resto features
X.columns = [f"f{i}" for i in range(X.shape[1])]  # Nombres genericos f0, f1, ...
print("X.shape", X.shape, "y.shape", y.shape)  # Shapes
print(y.value_counts())  # Clases
plt.figure(figsize=(4,3))  # Figura
y.value_counts().plot(kind="bar")  # Barras
plt.show()  # Grafico
num_x = X.select_dtypes(include=[np.number]).columns  # Numericas
if len(num_x) > 0:
    print("Top corr:", X[num_x].corrwith(y).abs().sort_values(ascending=False).head(5))  # Top 5
"""
        )
    )
    cells.append(md("## SECCIÓN 3"))
    cells.append(code(seccion3_enc()))
    cells.append(md("## SECCIÓN 4"))
    cells.append(code(seccion4_scaler()))
    cells.append(md("## SECCIÓN 5"))
    cells.append(code(seccion5_split(True)))
    cells.append(md("## SECCIÓN 6"))
    cells.append(
        code(
            """clf = LogisticRegression(max_iter=1000)  # Modelo
clf.fit(X_train, y_train)  # Entrenar
acc = accuracy_score(y_test, clf.predict(X_test))  # Accuracy
print("Accuracy:", acc)  # Resultado
score_aus = acc  # Score
"""
        )
    )
    cells.append(md("## SECCIÓN 7"))
    cells.append(
        code(
            f"""out = Path(r"{str(list(ROOT.glob('7. Statlog*'))[0]).replace(chr(92), '/')}")  # Carpeta
pd.concat([X_scaled_df, y.rename("label")], axis=1).to_csv(out / "australian_credit_clean.csv", index=False)  # Limpio
X_train.to_csv(out / "X_train.csv", index=False)  # Train
X_test.to_csv(out / "X_test.csv", index=False)  # Test
y_train.to_csv(out / "y_train.csv", index=True, header=True)  # y
y_test.to_csv(out / "y_test.csv", index=True, header=True)  # y test
print("Guardado australian_credit_clean.csv")  # Ok
"""
        )
    )
    cells.append(md("## SECCIÓN 8"))
    cells.append(
        code(
            """display(df_clean.head(10))  # Muestra
assert df_clean.isnull().sum().sum() == 0
print("RESUMEN Australian | Acc:", round(acc, 4))  # Resumen
"""
        )
    )
    return cells


def build_breast() -> list:
    folder = list(ROOT.glob("8. Breast*"))[0]
    csvp = folder / "breast+cancer+wisconsin+original" / "breast-cancer-wisconsin.data"
    cols = ["id", "clump", "cell_size", "cell_shape", "marginal", "epithelial", "bare_nuclei", "bland", "nucleoli", "mitoses", "class"]
    cells = nb_header(
        "Breast Cancer Wisconsin (Original)",
        "Medicina: mediciones de núcleos celulares para clasificar tumores (UCI).",
        "clasificación binaria",
        "`class`: 2 benigno, 4 maligno.",
    )
    cells.append(
        code(
            std_imports()
            + f"""
df = pd.read_csv(r"{str(csvp).replace(chr(92), '/')}", header=None, names={cols!r}, na_values="?")  # Cargar
display(df.head(10))  # Muestra
print(df.shape)  # Forma
display(df.describe(include="all").T)  # Stats
"""
        )
    )
    cells.append(md("## SECCIÓN 1"))
    cells.append(code(seccion1_template()))
    cells.append(md("## SECCIÓN 2"))
    cells.append(
        code(
            """y = (df_clean["class"] == 4).astype(int)  # 1 maligno
X = df_clean.drop(columns=["class"])  # Sin clase
if "id" in X.columns:
    X = X.drop(columns=["id"])  # Quitar ID
print("X.shape", X.shape, "y.shape", y.shape)  # Shapes
print(y.value_counts())  # Clases
plt.figure(figsize=(4,3))  # Figura
y.value_counts().plot(kind="bar")  # Barras
plt.show()  # Grafico
num_x = X.select_dtypes(include=[np.number]).columns  # Numericas
print("Top corr:", X[num_x].corrwith(y).abs().sort_values(ascending=False).head(5))  # Top 5
sns.heatmap(pd.concat([X[num_x[:5]], y.rename("y")], axis=1).corr(), annot=True)  # Heatmap
plt.show()
"""
        )
    )
    cells.append(md("## SECCIÓN 3 — Solo numéricas (sin one-hot obligatorio)"))
    cells.append(
        code(
            """X_enc = X.copy()  # Ya numericas
print("Columnas:", X_enc.columns.tolist())  # Lista
display(X_enc.head(5))  # Muestra
"""
        )
    )
    cells.append(md("## SECCIÓN 4"))
    cells.append(code(seccion4_scaler()))
    cells.append(md("## SECCIÓN 5"))
    cells.append(code(seccion5_split(True)))
    cells.append(md("## SECCIÓN 6"))
    cells.append(
        code(
            """clf = LogisticRegression(max_iter=1000)  # Modelo
clf.fit(X_train, y_train)  # Entrenar
acc = accuracy_score(y_test, clf.predict(X_test))  # Accuracy
print("Accuracy:", acc)  # Resultado
score_breast = acc  # Score
"""
        )
    )
    cells.append(md("## SECCIÓN 7"))
    cells.append(
        code(
            f"""out = Path(r"{str(list(ROOT.glob('8. Breast*'))[0]).replace(chr(92), '/')}")  # Carpeta
pd.concat([X_scaled_df, y.rename("malignant")], axis=1).to_csv(out / "breast_cancer_wisconsin_clean.csv", index=False)  # Limpio
X_train.to_csv(out / "X_train.csv", index=False)  # Train
X_test.to_csv(out / "X_test.csv", index=False)  # Test
y_train.to_csv(out / "y_train.csv", index=True, header=True)  # y
y_test.to_csv(out / "y_test.csv", index=True, header=True)  # y test
print("Guardado breast_cancer_wisconsin_clean.csv")  # Ok
"""
        )
    )
    cells.append(md("## SECCIÓN 8"))
    cells.append(
        code(
            """display(df_clean.head(10))  # Muestra
assert df_clean.isnull().sum().sum() == 0
print("RESUMEN Breast | Acc:", round(acc, 4))  # Resumen
"""
        )
    )
    return cells


def build_meningitis() -> list:
    folder = list(ROOT.glob("9. Meningitis*"))[0]
    csvp = folder / "archive" / "mening missing 12.csv"
    cells = nb_header(
        "Meningitis (Kaggle)",
        "Datos clínicos con **valores faltantes**; foco en imputación.",
        "clasificación multiclase",
        "`Diagnosis` — tipo (Viral/Bacterial/Unknown).",
    )
    cells.append(
        code(
            std_imports()
            + f"""
df = pd.read_csv(r"{str(csvp).replace(chr(92), '/')}")  # Cargar
display(df.head(10))  # Muestra
print(df.shape, df.dtypes)  # Info
display(df.describe(include="all").T)  # Stats
"""
        )
    )
    cells.append(md("## SECCIÓN 1"))
    cells.append(code(seccion1_template()))
    cells.append(md("## SECCIÓN 2"))
    cells.append(
        code(
            """y = df_clean["Diagnosis"]  # Objetivo multiclase
X = df_clean.drop(columns=["Diagnosis"])  # Features
if "Patient_ID" in X.columns:
    X = X.drop(columns=["Patient_ID"])  # Quitar ID
print("X.shape", X.shape, "y.shape", y.shape)  # Shapes
print(y.value_counts())  # Clases
plt.figure(figsize=(6,4))  # Figura
y.astype(str).value_counts().plot(kind="bar")  # Barras
plt.show()  # Grafico
num_x = X.select_dtypes(include=[np.number]).columns  # Numericas
le_y = LabelEncoder().fit_transform(y.astype(str))  # y numerico para corr
corr = X[num_x].corrwith(pd.Series(le_y, index=X.index)).abs().sort_values(ascending=False)  # Corr
print("Top 5 correlaciones:", corr.head(5))  # Top 5
"""
        )
    )
    cells.append(md("## SECCIÓN 3"))
    cells.append(code(seccion3_enc()))
    cells.append(md("## SECCIÓN 4"))
    cells.append(code(seccion4_scaler()))
    cells.append(md("## SECCIÓN 5"))
    cells.append(code(seccion5_split(True)))
    cells.append(md("## SECCIÓN 6"))
    cells.append(
        code(
            """clf = LogisticRegression(max_iter=1000)  # Multiclase
clf.fit(X_train, y_train)  # Entrenar
acc = accuracy_score(y_test, clf.predict(X_test))  # Accuracy
print("Accuracy:", acc)  # Resultado
score_men = acc  # Score
"""
        )
    )
    cells.append(md("## SECCIÓN 7"))
    cells.append(
        code(
            f"""out = Path(r"{str(list(ROOT.glob('9. Meningitis*'))[0]).replace(chr(92), '/')}")  # Carpeta
pd.concat([X_scaled_df, y.rename("Diagnosis")], axis=1).to_csv(out / "meningitis_clean.csv", index=False)  # Limpio
X_train.to_csv(out / "X_train.csv", index=False)  # Train
X_test.to_csv(out / "X_test.csv", index=False)  # Test
y_train.to_csv(out / "y_train.csv", index=True, header=True)  # y
y_test.to_csv(out / "y_test.csv", index=True, header=True)  # y test
print("Guardado meningitis_clean.csv")  # Ok
"""
        )
    )
    cells.append(md("## SECCIÓN 8"))
    cells.append(
        code(
            """display(df_clean.head(10))  # Muestra
assert df_clean.isnull().sum().sum() == 0
print("RESUMEN Meningitis | Acc:", round(acc, 4))  # Resumen
"""
        )
    )
    return cells


def main():
    nh = list(ROOT.glob("3. NHANES*"))[0]
    bk = list(ROOT.glob("4. Bike*"))[0]
    ad = list(ROOT.glob("5. Adult*"))[0]
    cr = list(ROOT.glob("6. Credit*"))[0]
    au = list(ROOT.glob("7. Statlog*"))[0]
    br = list(ROOT.glob("8. Breast*"))[0]
    me = list(ROOT.glob("9. Meningitis*"))[0]
    save_nb(nh / "nhanes_ml.ipynb", build_nhanes())
    save_nb(bk / "bike_sharing_ml.ipynb", build_bike())
    save_nb(ad / "adult_census_ml.ipynb", build_adult())
    save_nb(cr / "credit_approval_ml.ipynb", build_credit())
    save_nb(au / "australian_credit_ml.ipynb", build_australian())
    save_nb(br / "breast_cancer_wisconsin_ml.ipynb", build_breast())
    save_nb(me / "meningitis_ml.ipynb", build_meningitis())
    print("Notebooks 3-9 generados OK")


if __name__ == "__main__":
    main()
