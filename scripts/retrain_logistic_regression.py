import os
import re
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/clean_training_data.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

required_columns = [
    "Resume Text",
    "Education",
    "Experience Years",
    "Skills",
    "Job Role",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["Resume Text"] = df["Resume Text"].fillna("").astype(str)
df["Education"] = df["Education"].fillna("Unknown").astype(str)
df["Skills"] = df["Skills"].fillna("").astype(str)
df["Experience Years"] = pd.to_numeric(
    df["Experience Years"],
    errors="coerce"
).fillna(0)

df["Job Role"] = df["Job Role"].fillna("").astype(str)

print(f"Records: {len(df)}")
print(f"Unique roles: {df['Job Role'].nunique()}")


# ============================================================
# CLEAN TEXT
# ============================================================

print("\nCleaning text...")

df["Resume Clean"] = df["Resume Text"].apply(clean_text)
df["Skills Clean"] = df["Skills"].apply(clean_text)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df[
        [
            "Resume Clean",
            "Skills Clean",
            "Education",
            "Experience Years",
        ]
    ],
    df["Job Role"],
    test_size=0.20,
    random_state=42,
    stratify=df["Job Role"],
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# ============================================================
# RESUME TF-IDF
# ============================================================

print("\nCreating Resume TF-IDF...")

resume_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
)

X_train_resume = resume_vectorizer.fit_transform(
    X_train["Resume Clean"]
)

X_test_resume = resume_vectorizer.transform(
    X_test["Resume Clean"]
)

print("Resume features:", X_train_resume.shape[1])


# ============================================================
# SKILLS TF-IDF
# ============================================================

print("\nCreating Skills TF-IDF...")

skills_vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
)

X_train_skills = skills_vectorizer.fit_transform(
    X_train["Skills Clean"]
)

X_test_skills = skills_vectorizer.transform(
    X_test["Skills Clean"]
)

print("Skills features:", X_train_skills.shape[1])


# ============================================================
# EDUCATION ENCODER
# ============================================================

print("\nEncoding education...")

education_encoder = OneHotEncoder(
    handle_unknown="ignore"
)

X_train_education = education_encoder.fit_transform(
    X_train[["Education"]]
)

X_test_education = education_encoder.transform(
    X_test[["Education"]]
)

print("Education features:", X_train_education.shape[1])


# ============================================================
# EXPERIENCE
# ============================================================

X_train_experience = X_train[
    ["Experience Years"]
].values

X_test_experience = X_test[
    ["Experience Years"]
].values


# ============================================================
# COMBINE FEATURES
# ============================================================

print("\nCombining features...")

X_train_final = hstack(
    [
        X_train_resume,
        X_train_skills,
        X_train_education,
        X_train_experience,
    ]
).tocsr()

X_test_final = hstack(
    [
        X_test_resume,
        X_test_skills,
        X_test_education,
        X_test_experience,
    ]
).tocsr()

print("Final training features:", X_train_final.shape)
print("Final testing features:", X_test_final.shape)


# ============================================================
# LABEL ENCODING
# ============================================================

print("\nEncoding job roles...")

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Number of classes:", len(label_encoder.classes_))


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")
print("This may take some time...")

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    solver="lbfgs",
)

model.fit(
    X_train_final,
    y_train_encoded
)


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(X_test_final)

accuracy = accuracy_score(
    y_test_encoded,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Model Classes: {len(model.classes_)}")
print(f"Model Features: {model.n_features_in_}")


# ============================================================
# SAVE ARTIFACTS
# ============================================================

print("\nSaving model artifacts...")

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "logistic_regression_model.pkl"
    )
)

joblib.dump(
    resume_vectorizer,
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)

joblib.dump(
    skills_vectorizer,
    os.path.join(
        MODEL_DIR,
        "skills_tfidf.pkl"
    )
)

joblib.dump(
    education_encoder,
    os.path.join(
        MODEL_DIR,
        "education_onehot_encoder.pkl"
    )
)

joblib.dump(
    label_encoder,
    os.path.join(
        MODEL_DIR,
        "job_role_label_encoder.pkl"
    )
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    MODEL_DIR,
    "logistic_regression_training_report.txt"
)

with open(report_path, "w", encoding="utf-8") as f:

    f.write("LOGISTIC REGRESSION TRAINING REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset records: {len(df)}\n")
    f.write(
        f"Unique job roles: {df['Job Role'].nunique()}\n"
    )

    f.write(
        f"Training records: {len(X_train)}\n"
    )

    f.write(
        f"Testing records: {len(X_test)}\n"
    )

    f.write(
        f"Resume TF-IDF features: "
        f"{X_train_resume.shape[1]}\n"
    )

    f.write(
        f"Skills TF-IDF features: "
        f"{X_train_skills.shape[1]}\n"
    )

    f.write(
        f"Education features: "
        f"{X_train_education.shape[1]}\n"
    )

    f.write(
        f"Total model features: "
        f"{X_train_final.shape[1]}\n"
    )

    f.write(
        f"Model classes: "
        f"{len(model.classes_)}\n"
    )

    f.write(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write("\nJOB ROLE CLASSES\n")
    f.write("-" * 60 + "\n")

    for i, role in enumerate(label_encoder.classes_):
        f.write(f"{i}: {role}\n")


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nSaved files:")

print("✓ models/logistic_regression_model.pkl")
print("✓ models/tfidf_vectorizer.pkl")
print("✓ models/skills_tfidf.pkl")
print("✓ models/education_onehot_encoder.pkl")
print("✓ models/job_role_label_encoder.pkl")
print("✓ models/logistic_regression_training_report.txt")

print("\nFinal model:")
print("Features:", model.n_features_in_)
print("Classes:", len(model.classes_))
print(f"Accuracy: {accuracy * 100:.2f}%")