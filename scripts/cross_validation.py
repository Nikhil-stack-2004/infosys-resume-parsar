import os
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clean_training_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

column_map = {}

for col in data.columns:

    normalized = col.lower().strip()

    if normalized in ["resume text", "resume_text"]:
        column_map[col] = "Resume Text"

    elif normalized in ["skills", "skill"]:
        column_map[col] = "Skills"

    elif normalized == "education":
        column_map[col] = "Education"

    elif normalized in [
        "experience years",
        "experience_years",
        "experience"
    ]:
        column_map[col] = "Experience Years"

    elif normalized in [
        "job role",
        "job_role",
        "role"
    ]:
        column_map[col] = "Job Role"

data = data.rename(columns=column_map)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


data["Resume Text"] = (
    data["Resume Text"]
    .fillna("")
    .apply(clean_text)
)

data["Skills"] = (
    data["Skills"]
    .fillna("")
    .apply(clean_text)
)

data["Education"] = (
    data["Education"]
    .fillna("unknown")
    .astype(str)
    .str.lower()
)

data["Experience Years"] = (
    pd.to_numeric(
        data["Experience Years"],
        errors="coerce"
    )
    .fillna(0)
)


# ============================================================
# FEATURES
# ============================================================

X = data[
    [
        "Resume Text",
        "Skills",
        "Education",
        "Experience Years"
    ]
]

y = data["Job Role"].astype(str)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# RESUME TF-IDF
# ============================================================

print("Creating Resume TF-IDF...")

resume_tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_resume = resume_tfidf.fit_transform(
    X_train["Resume Text"]
)


# ============================================================
# SKILLS TF-IDF
# ============================================================

print("Creating Skills TF-IDF...")

skills_tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_skills = skills_tfidf.fit_transform(
    X_train["Skills"]
)


# ============================================================
# EDUCATION
# ============================================================

education_encoder = OneHotEncoder(
    handle_unknown="ignore"
)

X_train_education = education_encoder.fit_transform(
    X_train[["Education"]]
)


# ============================================================
# EXPERIENCE
# ============================================================

X_train_experience = (
    X_train[["Experience Years"]].values
)


# ============================================================
# COMBINE FEATURES
# ============================================================

X_train_final = hstack([
    X_train_resume,
    X_train_skills,
    X_train_education,
    X_train_experience
]).tocsr()


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train
)


print("Final feature shape:", X_train_final.shape)
print("Number of classes:", len(label_encoder.classes_))


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print("\nRunning 5-Fold Cross-Validation...")
print("This may take some time.\n")

cv_scores = cross_val_score(
    model,
    X_train_final,
    y_train_encoded,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


# ============================================================
# RESULTS
# ============================================================

print("\n====================================")
print("5-FOLD CROSS-VALIDATION RESULTS")
print("====================================")

for i, score in enumerate(cv_scores, start=1):

    print(
        f"Fold {i}: "
        f"{score:.4f} "
        f"({score * 100:.2f}%)"
    )


print(
    "\nMean CV Accuracy:",
    round(cv_scores.mean(), 4)
)

print(
    "Mean CV Accuracy (%):",
    round(cv_scores.mean() * 100, 2)
)

print(
    "Standard Deviation:",
    round(cv_scores.std(), 4)
)

print(
    "Standard Deviation (%):",
    round(cv_scores.std() * 100, 2)
)

print("\nCross-validation completed successfully!")