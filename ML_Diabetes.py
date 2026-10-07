# ============================================================
# EDA + PREPROCESSING + MACHINE LEARNING
# DATASET : DIABETIC DATA
# Target  : readmitted
# Struktur dibuat mengikuti pola EDA NYC Airbnb yang digunakan
# sebagai acuan: Data Understanding -> Data Quality -> EDA
# -> Anomaly/Preprocessing -> Modeling -> Comparison -> Conclusion
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. KONFIGURASI
# ============================================================

FILE_PATH = "diabetic_data.csv"
OUTPUT_DIR = "hasil_eda_diabetic"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 80)
print("DIABETIC DATA - EDA + MACHINE LEARNING")
print("=" * 80)

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print("\nERROR: File dataset tidak ditemukan!")
    print(f"Pastikan file '{FILE_PATH}' berada satu folder dengan program.")
    raise SystemExit

print("\nDataset berhasil dibaca.")
print(f"Nama file    : {FILE_PATH}")
print(f"Jumlah baris : {df.shape[0]:,}")
print(f"Jumlah kolom : {df.shape[1]}")


# ============================================================
# 1. DATA UNDERSTANDING
# ============================================================

print("\n\n" + "=" * 80)
print("1. DATA UNDERSTANDING")
print("=" * 80)

print("\n1.1 JUMLAH BARIS DAN KOLOM")
print("-" * 80)
print(f"Jumlah baris  : {df.shape[0]:,}")
print(f"Jumlah kolom  : {df.shape[1]}")

print("\n1.2 LIMA DATA PERTAMA")
print("-" * 80)
print(df.head())

print("\n1.3 LIMA DATA TERAKHIR")
print("-" * 80)
print(df.tail())

print("\n1.4 INFORMASI DATASET")
print("-" * 80)
df.info()

print("\n1.5 TIPE DATA SETIAP VARIABEL")
print("-" * 80)

dtype_table = pd.DataFrame({
    "Kolom": df.columns,
    "Tipe Data": df.dtypes.astype(str).values,
    "Jumlah Nilai": df.count().values,
    "Missing": df.isnull().sum().values
})

print(dtype_table.to_string(index=False))

# Deskripsi sederhana berdasarkan struktur dataset
print("\n1.6 DAFTAR VARIABEL")
print("-" * 80)
for i, kolom in enumerate(df.columns, start=1):
    print(f"{i:02d}. {kolom}")


# ============================================================
# 2. DATA QUALITY
# ============================================================

print("\n\n" + "=" * 80)
print("2. DATA QUALITY")
print("=" * 80)

# ------------------------------------------------------------
# 2.1 Missing Value
# ------------------------------------------------------------

print("\n2.1 MISSING VALUE")
print("-" * 80)

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "Kolom": df.columns,
    "Missing": missing_count.values,
    "Persentase (%)": missing_percent.round(2).values
}).sort_values("Missing", ascending=False)

print(missing_table.to_string(index=False))

total_missing = df.isnull().sum().sum()
print(f"\nTotal missing value : {total_missing:,}")

# ------------------------------------------------------------
# 2.2 Duplicate
# ------------------------------------------------------------

print("\n2.2 DUPLICATE")
print("-" * 80)

duplicate_count = df.duplicated().sum()
print(f"Jumlah data duplikat : {duplicate_count:,}")

# ------------------------------------------------------------
# 2.3 Target Distribution
# ------------------------------------------------------------

print("\n2.3 DISTRIBUSI TARGET - readmitted")
print("-" * 80)

target_counts = df["readmitted"].value_counts(dropna=False)
target_percent = df["readmitted"].value_counts(normalize=True, dropna=False) * 100

target_table = pd.DataFrame({
    "Jumlah": target_counts,
    "Persentase (%)": target_percent.round(2)
})

print(target_table)

# ------------------------------------------------------------
# 2.4 Nilai '?' yang digunakan dataset
# ------------------------------------------------------------

print("\n2.4 NILAI '?' / UNKNOWN")
print("-" * 80)

question_mark_table = pd.DataFrame({
    "Kolom": df.columns,
    "Jumlah '?'": [(df[col].astype(str) == "?").sum() for col in df.columns]
})

question_mark_table = question_mark_table[
    question_mark_table["Jumlah '?'"] > 0
].sort_values("Jumlah '?'", ascending=False)

print(question_mark_table.to_string(index=False))

total_question_mark = sum(
    (df[col].astype(str) == "?").sum() for col in df.columns
)

print(f"\nTotal nilai '?' : {total_question_mark:,}")


# ============================================================
# 3. EDA
# ============================================================

print("\n\n" + "=" * 80)
print("3. EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# ------------------------------------------------------------
# 3.1 Statistik Deskriptif
# ------------------------------------------------------------

print("\n3.1 STATISTIK DESKRIPTIF")
print("-" * 80)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

descriptive_stats = df[numeric_columns].describe().T
descriptive_stats["median"] = df[numeric_columns].median()

print(descriptive_stats)

descriptive_stats.to_csv(
    os.path.join(OUTPUT_DIR, "statistik_deskriptif.csv")
)

# ------------------------------------------------------------
# 3.2 Distribusi Target
# ------------------------------------------------------------

print("\n3.2 BAR CHART READMITTED")

plt.figure(figsize=(9, 6))
sns.countplot(data=df, x="readmitted", order=df["readmitted"].value_counts().index)
plt.title("Distribusi Target Readmitted")
plt.xlabel("Status Readmitted")
plt.ylabel("Jumlah Pasien")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "01_distribusi_readmitted.png"),
    dpi=300
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 3.3 Distribusi Umur
# ------------------------------------------------------------

print("3.3 BAR CHART AGE")

age_counts = df["age"].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=age_counts.index, y=age_counts.values)
plt.title("Distribusi Kelompok Umur Pasien")
plt.xlabel("Kelompok Umur")
plt.ylabel("Jumlah Pasien")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "02_distribusi_age.png"),
    dpi=300
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 3.4 Distribusi Gender
# ------------------------------------------------------------

print("3.4 BAR CHART GENDER")

gender_counts = df["gender"].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=gender_counts.index, y=gender_counts.values)
plt.title("Distribusi Gender")
plt.xlabel("Gender")
plt.ylabel("Jumlah Pasien")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "03_distribusi_gender.png"),
    dpi=300
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 3.5 Time in Hospital
# ------------------------------------------------------------

print("3.5 HISTOGRAM TIME IN HOSPITAL")

plt.figure(figsize=(10, 6))
sns.histplot(df["time_in_hospital"], bins=14, kde=True)
plt.title("Distribusi Lama Rawat Inap")
plt.xlabel("Time in Hospital")
plt.ylabel("Jumlah Pasien")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "04_histogram_time_in_hospital.png"),
    dpi=300
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 3.6 Number of Medications
# ------------------------------------------------------------

print("3.6 HISTOGRAM NUMBER OF MEDICATIONS")

plt.figure(figsize=(10, 6))
sns.histplot(df["num_medications"], bins=30, kde=True)
plt.title("Distribusi Jumlah Obat")
plt.xlabel("Number of Medications")
plt.ylabel("Jumlah Pasien")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "05_histogram_num_medications.png"),
    dpi=300
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 3.7 Boxplot Time in Hospital
# ------------------------------------------------------------

print("3.7 BOXPLOT TIME IN HOSPITAL")

plt.figure(figsize=(10, 5))
sns.boxplot(x=df["time_in_hospital"])
plt.title("Boxplot Time in Hospital")
plt.xlabel("Time in Hospital")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "06_boxplot_time_in_hospital.png"),
    dpi=300
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 3.8 Hubungan Time in Hospital dengan Readmitted
# ------------------------------------------------------------

print("3.8 BOXPLOT TIME IN HOSPITAL VS READMITTED")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="readmitted", y="time_in_hospital")
plt.title("Time in Hospital Berdasarkan Status Readmitted")
plt.xlabel("Status Readmitted")
plt.ylabel("Time in Hospital")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "07_boxplot_time_vs_readmitted.png"),
    dpi=300
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 3.9 Hubungan Age dengan Readmitted
# ------------------------------------------------------------

print("3.9 BAR CHART AGE VS READMITTED")

age_readmitted = pd.crosstab(
    df["age"],
    df["readmitted"],
    normalize="index"
) * 100

age_readmitted = age_readmitted.reindex(
    df["age"].value_counts().sort_index().index
)

age_readmitted.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Persentase Readmitted Berdasarkan Kelompok Umur")
plt.xlabel("Kelompok Umur")
plt.ylabel("Persentase (%)")
plt.xticks(rotation=45)
plt.legend(title="Readmitted")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "08_age_vs_readmitted.png"),
    dpi=300
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 3.10 Korelasi Variabel Numerik
# ------------------------------------------------------------

print("3.10 CORRELATION HEATMAP")

correlation_matrix = df[numeric_columns].corr()

print("\nMatriks korelasi:")
print(correlation_matrix.round(3))

plt.figure(figsize=(14, 11))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Correlation Heatmap Variabel Numerik")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "09_correlation_heatmap.png"),
    dpi=300
)
plt.show()
plt.close()


# ============================================================
# 4. ANOMALY DETECTION
# ============================================================

print("\n\n" + "=" * 80)
print("4. ANOMALY DETECTION")
print("=" * 80)

# ------------------------------------------------------------
# 4.1 Outlier Time in Hospital dengan IQR
# ------------------------------------------------------------

print("\n4.1 OUTLIER TIME IN HOSPITAL")
print("-" * 80)

Q1 = df["time_in_hospital"].quantile(0.25)
Q3 = df["time_in_hospital"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_time = df[
    (df["time_in_hospital"] < lower_bound) |
    (df["time_in_hospital"] > upper_bound)
]

print(f"Q1              : {Q1:.2f}")
print(f"Q3              : {Q3:.2f}")
print(f"IQR             : {IQR:.2f}")
print(f"Batas bawah     : {lower_bound:.2f}")
print(f"Batas atas      : {upper_bound:.2f}")
print(f"Jumlah outlier  : {len(outliers_time):,}")

# ------------------------------------------------------------
# 4.2 Cek nilai negatif pada variabel numerik
# ------------------------------------------------------------

print("\n4.2 NILAI NEGATIF")
print("-" * 80)

negative_table = []

for col in numeric_columns:
    count_negative = (df[col] < 0).sum()
    negative_table.append([col, count_negative])

negative_table = pd.DataFrame(
    negative_table,
    columns=["Kolom", "Jumlah Negatif"]
)

print(
    negative_table[
        negative_table["Jumlah Negatif"] > 0
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 4.3 Missing dan '?' sebagai masalah kualitas data
# ------------------------------------------------------------

print("\n4.3 KOLOM DENGAN MISSING / '?' TINGGI")
print("-" * 80)

quality_table = pd.DataFrame({
    "Kolom": df.columns,
    "Missing": df.isnull().sum().values,
    "Missing_%": (df.isnull().mean() * 100).round(2).values,
    "Question_Mark": [
        (df[col].astype(str) == "?").sum()
        for col in df.columns
    ]
})

quality_table["Question_Mark_%"] = (
    quality_table["Question_Mark"] / len(df) * 100
).round(2)

print(
    quality_table.sort_values(
        ["Question_Mark", "Missing"],
        ascending=False
    ).head(20).to_string(index=False)
)


# ============================================================
# 5. PREPROCESSING MACHINE LEARNING
# ============================================================

print("\n\n" + "=" * 80)
print("5. PREPROCESSING MACHINE LEARNING")
print("=" * 80)

# ------------------------------------------------------------
# 5.1 Membersihkan simbol '?'
# ------------------------------------------------------------

ml_df = df.copy()

# Dataset menggunakan '?' sebagai representasi missing value.
ml_df = ml_df.replace("?", np.nan)

print("\n5.1 Nilai '?' telah diubah menjadi NaN.")

# ------------------------------------------------------------
# 5.2 Menghapus duplicate
# ------------------------------------------------------------

before_duplicate = len(ml_df)
ml_df = ml_df.drop_duplicates()
after_duplicate = len(ml_df)

print(f"Data sebelum hapus duplicate : {before_duplicate:,}")
print(f"Data setelah hapus duplicate : {after_duplicate:,}")
print(f"Duplicate yang dihapus       : {before_duplicate - after_duplicate:,}")

# ------------------------------------------------------------
# 5.3 Feature engineering sederhana untuk diagnosis
# ------------------------------------------------------------

# diag_1, diag_2, diag_3 merupakan kode diagnosis.
# Agar tidak membuat terlalu banyak kategori unik, digunakan
# 3 karakter pertama sebagai kelompok diagnosis.

for col in ["diag_1", "diag_2", "diag_3"]:
    if col in ml_df.columns:
        ml_df[col] = (
            ml_df[col]
            .astype("string")
            .str.replace(r"^V", "V", regex=True)
            .str.replace(r"^E", "E", regex=True)
            .str[:3]
        )

# IMPORTANT:
# .astype("string") menggunakan pandas.NA sebagai missing value.
# SimpleImputer pada beberapa versi scikit-learn tidak dapat
# menangani pandas.NA pada kolom object. Karena itu seluruh dataframe
# ML dinormalisasi ke object/NumPy NaN sebelum masuk ke sklearn.
for col in ml_df.columns:
    if pd.api.types.is_object_dtype(ml_df[col]) or pd.api.types.is_string_dtype(ml_df[col]):
        ml_df[col] = ml_df[col].astype(object)
        ml_df[col] = ml_df[col].where(pd.notna(ml_df[col]), np.nan)

# ------------------------------------------------------------
# 5.4 Menghapus kolom identifier dan target
# ------------------------------------------------------------

TARGET = "readmitted"

# ID bukan karakteristik klinis dan tidak digunakan sebagai fitur.
# encounter_id dan patient_nbr juga berpotensi membuat model
# mempelajari identitas/urutan data, bukan pola fitur.
DROP_COLUMNS = [
    "encounter_id",
    "patient_nbr"
]

# weight memiliki missing value sangat tinggi pada dataset ini.
# Untuk menghindari fitur yang sangat sparse, kolom tersebut
# tidak digunakan dalam model.
if "weight" in ml_df.columns:
    DROP_COLUMNS.append("weight")

X = ml_df.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")
y = ml_df[TARGET].copy()

print("\n5.4 KOLOM YANG DIHAPUS DARI MODEL")
print(DROP_COLUMNS)

print("\nTarget:")
print(y.value_counts())

# ------------------------------------------------------------
# 5.5 Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n5.5 TRAIN TEST SPLIT")
print("-" * 80)
print(f"Data training : {X_train.shape[0]:,}")
print(f"Data testing  : {X_test.shape[0]:,}")
print(f"Jumlah fitur  : {X_train.shape[1]}")

# ------------------------------------------------------------
# 5.6 Identifikasi numerik dan kategorikal
# ------------------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

# Pastikan kolom kategorikal benar-benar menggunakan object biasa
# dan missing value berupa np.nan, bukan pandas.NA.
for col in categorical_features:
    X_train[col] = X_train[col].astype(object)
    X_train[col] = X_train[col].where(pd.notna(X_train[col]), np.nan)
    X_test[col] = X_test[col].astype(object)
    X_test[col] = X_test[col].where(pd.notna(X_test[col]), np.nan)

print("\nFitur numerik:")
print(numeric_features)

print("\nFitur kategorikal:")
print(categorical_features)

# ------------------------------------------------------------
# 5.7 Pipeline preprocessing
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", missing_values=np.nan)),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\n5.7 PREPROCESSING")
print("-" * 80)
print("Numerik     : median imputation + StandardScaler")
print("Kategorikal : most-frequent imputation + OneHotEncoder")
print("Unknown     : handle_unknown='ignore'")
print("Kategori langka digabung dengan min_frequency=10")


# ============================================================
# 6. MACHINE LEARNING - 3 MODEL
# ============================================================

print("\n\n" + "=" * 80)
print("6. MACHINE LEARNING")
print("=" * 80)

# Model 1: Logistic Regression
model_logistic = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="saga",
                random_state=RANDOM_STATE
            )
        )
    ]
)

# Model 2: Random Forest
model_random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )
    ]
)

# Model 3: Extra Trees
model_extra_trees = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=25,
                min_samples_split=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )
    ]
)

models = {
    "Logistic Regression": model_logistic,
    "Random Forest": model_random_forest,
    "Extra Trees": model_extra_trees
}


# ============================================================
# 7. TRAINING DAN EVALUASI MODEL
# ============================================================

print("\n\n" + "=" * 80)
print("7. TRAINING DAN EVALUASI MODEL")
print("=" * 80)

results = []
predictions = {}

for model_name, model in models.items():

    print(f"\n{'-' * 80}")
    print(f"TRAINING: {model_name}")
    print(f"{'-' * 80}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions[model_name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision_Weighted": precision,
        "Recall_Weighted": recall,
        "F1_Weighted": f1,
        "Precision_Macro": macro_precision,
        "Recall_Macro": macro_recall,
        "F1_Macro": macro_f1
    })

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision Weighted : {precision:.4f}")
    print(f"Recall Weighted    : {recall:.4f}")
    print(f"F1 Weighted        : {f1:.4f}")
    print(f"Precision Macro    : {macro_precision:.4f}")
    print(f"Recall Macro       : {macro_recall:.4f}")
    print(f"F1 Macro           : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Confusion Matrix
    labels = sorted(y_test.dropna().unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"10_confusion_matrix_{safe_name}.png"
        ),
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 8. PERBANDINGAN MODEL
# ============================================================

print("\n\n" + "=" * 80)
print("8. PERBANDINGAN MODEL")
print("=" * 80)

results_df = pd.DataFrame(results)

# Urutan kolom
results_df = results_df[
    [
        "Model",
        "Accuracy",
        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",
        "Precision_Macro",
        "Recall_Macro",
        "F1_Macro"
    ]
]

print("\nTABEL PERBANDINGAN:")
print(
    results_df.round(4).to_string(index=False)
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "perbandingan_model.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 8.1 Visualisasi perbandingan
# ------------------------------------------------------------

metrics_to_plot = [
    "Accuracy",
    "Precision_Weighted",
    "Recall_Weighted",
    "F1_Weighted"
]

comparison_plot = results_df.set_index("Model")[metrics_to_plot]

plt.figure(figsize=(13, 7))
comparison_plot.plot(
    kind="bar",
    figsize=(13, 7)
)

plt.title("Perbandingan Performa Model")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "11_perbandingan_performa_model.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 9. KESIMPULAN OTOMATIS
# ============================================================

print("\n\n" + "=" * 80)
print("9. KESIMPULAN PERBANDINGAN MODEL")
print("=" * 80)

# Nilai terbesar untuk tiap metrik hanya digunakan untuk
# membuat ringkasan faktual dari hasil eksperimen.
metric_names = [
    "Accuracy",
    "Precision_Weighted",
    "Recall_Weighted",
    "F1_Weighted",
    "F1_Macro"
]

print("\nHASIL TERTINGGI BERDASARKAN SETIAP METRIK:")
print("-" * 80)

for metric in metric_names:
    idx = results_df[metric].idxmax()
    model_name = results_df.loc[idx, "Model"]
    score = results_df.loc[idx, metric]

    print(
        f"{metric:20s}: "
        f"{model_name:22s} = {score:.4f}"
    )

print("\nINTERPRETASI:")
print("-" * 80)
print(
    "1. Accuracy menunjukkan proporsi seluruh prediksi yang benar."
)
print(
    "2. Precision Weighted menunjukkan ketepatan prediksi dengan "
    "memperhitungkan proporsi masing-masing kelas."
)
print(
    "3. Recall Weighted menunjukkan kemampuan model menemukan "
    "data dari setiap kelas dengan bobot berdasarkan jumlah kelas."
)
print(
    "4. F1 Weighted merupakan gabungan precision dan recall "
    "dengan bobot berdasarkan jumlah sampel setiap kelas."
)
print(
    "5. F1 Macro menghitung F1 setiap kelas secara seimbang, "
    "sehingga berguna untuk melihat performa pada kelas yang "
    "jumlah datanya lebih sedikit."
)

# ------------------------------------------------------------
# 9.1 Kesimpulan berbasis F1 Macro
# ------------------------------------------------------------

best_macro_idx = results_df["F1_Macro"].idxmax()
best_macro_model = results_df.loc[best_macro_idx, "Model"]
best_macro_score = results_df.loc[best_macro_idx, "F1_Macro"]

print("\nKESIMPULAN BERDASARKAN HASIL EKSPERIMEN:")
print("-" * 80)
print(
    f"Pada data testing, nilai F1 Macro tertinggi dalam eksperimen "
    f"ini adalah {best_macro_model} dengan skor {best_macro_score:.4f}."
)
print(
    "F1 Macro dipakai sebagai salah satu acuan utama karena dataset "
    "memiliki lebih dari satu kelas pada target readmitted."
)
print(
    "Namun, pemilihan model akhir sebaiknya mempertimbangkan tujuan "
    "analisis dan metrik yang dianggap paling penting."
)


# ============================================================
# 10. SIMPAN LAPORAN RINGKAS
# ============================================================

print("\n\n" + "=" * 80)
print("10. MENYIMPAN LAPORAN")
print("=" * 80)

report_path = os.path.join(
    OUTPUT_DIR,
    "laporan_eda_machine_learning.txt"
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("LAPORAN EDA + MACHINE LEARNING\n")
    file.write("DIABETIC DATASET\n")
    file.write("=" * 80 + "\n\n")

    file.write("1. DATA UNDERSTANDING\n")
    file.write("-" * 80 + "\n")
    file.write(f"Jumlah baris  : {len(df):,}\n")
    file.write(f"Jumlah kolom  : {len(df.columns)}\n\n")
    file.write("Kolom dataset:\n")

    for kolom in df.columns:
        file.write(f"- {kolom}\n")

    file.write("\n\n2. DATA QUALITY\n")
    file.write("-" * 80 + "\n")
    file.write(f"Total missing value : {total_missing:,}\n")
    file.write(f"Total duplicate     : {duplicate_count:,}\n")
    file.write(f"Total '?'           : {total_question_mark:,}\n")

    file.write("\nDistribusi target:\n")
    file.write(target_table.to_string())
    file.write("\n")

    file.write("\n\n3. PREPROCESSING\n")
    file.write("-" * 80 + "\n")
    file.write(
        "1. Mengubah '?' menjadi NaN.\n"
        "2. Menghapus duplicate.\n"
        "3. Menggunakan tiga karakter awal diagnosis sebagai kelompok kategori.\n"
        "4. Menghapus encounter_id, patient_nbr, dan weight dari fitur model.\n"
        "5. Numerik: median imputation + StandardScaler.\n"
        "6. Kategorikal: most-frequent imputation + OneHotEncoder.\n"
        "7. Train-test split 80:20 dengan stratifikasi target.\n"
    )

    file.write("\n\n4. MODEL\n")
    file.write("-" * 80 + "\n")
    file.write(
        "1. Logistic Regression\n"
        "2. Random Forest\n"
        "3. Extra Trees\n"
    )

    file.write("\n\n5. PERBANDINGAN MODEL\n")
    file.write("-" * 80 + "\n")
    file.write(results_df.round(4).to_string(index=False))
    file.write("\n")

    file.write("\n\n6. HASIL TERTINGGI PER METRIK\n")
    file.write("-" * 80 + "\n")

    for metric in metric_names:
        idx = results_df[metric].idxmax()
        model_name = results_df.loc[idx, "Model"]
        score = results_df.loc[idx, metric]

        file.write(
            f"{metric}: {model_name} = {score:.4f}\n"
        )

    file.write("\n\n7. KESIMPULAN\n")
    file.write("-" * 80 + "\n")
    file.write(
        f"F1 Macro tertinggi pada data testing diperoleh oleh "
        f"{best_macro_model} dengan skor {best_macro_score:.4f}.\n"
    )
    file.write(
        "F1 Macro digunakan sebagai salah satu acuan karena target "
        "readmitted memiliki beberapa kelas.\n"
    )
    file.write(
        "Kesimpulan model tetap perlu mempertimbangkan tujuan analisis "
        "dan metrik evaluasi yang diprioritaskan.\n"
    )

print(f"\nLaporan berhasil disimpan:")
print(f"  {report_path}")

print("\nSemua grafik dan hasil perbandingan tersimpan di folder:")
print(f"  {OUTPUT_DIR}/")

print("\n" + "=" * 80)
print("EDA + MACHINE LEARNING SELESAI")
print("=" * 80)
