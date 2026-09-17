import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack


# ============================================================
# 1. PATHS
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
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("HYPERPARAMETER TUNING - LOGISTIC REGRESSION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. SELECT REQUIRED COLUMNS
# ============================================================

df = df[
    [
        "Resume Text",
        "Education",
        "Experience Years",
        "Skills",
        "Job Role"
    ]
].copy()


# ============================================================
# 4. CLEAN TEXT
# ============================================================

def clean_text(text):
    text = str(text).lower()

    import re

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["Resume Text"] = df["Resume Text"].apply(clean_text)
df["Skills"] = df["Skills"].apply(clean_text)
df["Education"] = df["Education"].fillna("").astype(str).str.lower()

df["Experience Years"] = pd.to_numeric(
    df["Experience Years"],
    errors="coerce"
).fillna(0)


# ============================================================
# 5. COMBINE TEXT
# ============================================================

df["Combined Text"] = (
    df["Resume Text"]
    + " "
    + df["Skills"]
)


# ============================================================
# 6. FEATURES
# ============================================================

X_text = df["Combined Text"]

X_education = df[["Education"]]

X_experience = df[
    ["Experience Years"]
]

y = df["Job Role"]


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train_text, X_test_text, \
X_train_education, X_test_education, \
X_train_experience, X_test_experience, \
y_train, y_test = train_test_split(
    X_text,
    X_education,
    X_experience,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. RESUME + SKILLS TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

resume_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

skills_vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_resume = resume_vectorizer.fit_transform(
    X_train_text
)

X_test_resume = resume_vectorizer.transform(
    X_test_text
)

X_train_skills = skills_vectorizer.fit_transform(
    X_train_text
)

X_test_skills = skills_vectorizer.transform(
    X_test_text
)


# ============================================================
# 9. EDUCATION ENCODING
# ============================================================

education_encoder = OneHotEncoder(
    handle_unknown="ignore"
)

X_train_education_encoded = (
    education_encoder.fit_transform(
        X_train_education
    )
)

X_test_education_encoded = (
    education_encoder.transform(
        X_test_education
    )
)


# ============================================================
# 10. EXPERIENCE
# ============================================================

X_train_experience_values = (
    X_train_experience.values
)

X_test_experience_values = (
    X_test_experience.values
)


# ============================================================
# 11. COMBINE ALL FEATURES
# ============================================================

X_train = hstack(
    [
        X_train_resume,
        X_train_skills,
        X_train_experience_values,
        X_train_education_encoded
    ]
)

X_test = hstack(
    [
        X_test_resume,
        X_test_skills,
        X_test_experience_values,
        X_test_education_encoded
    ]
)


# ============================================================
# 12. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train
)

y_test_encoded = label_encoder.transform(
    y_test
)


print("\nFeature preparation completed.")

print(
    "Training features:",
    X_train.shape
)

print(
    "Testing features:",
    X_test.shape
)

print(
    "Number of classes:",
    len(label_encoder.classes_)
)


# ============================================================
# 13. BASELINE MODEL
# ============================================================

print("\nTraining baseline Logistic Regression...")

baseline_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

baseline_model.fit(
    X_train,
    y_train_encoded
)

baseline_accuracy = (
    baseline_model.score(
        X_test,
        y_test_encoded
    )
)

print(
    f"Baseline Test Accuracy: "
    f"{baseline_accuracy * 100:.2f}%"
)


# ============================================================
# 14. HYPERPARAMETER GRID
# ============================================================

print("\nStarting hyperparameter tuning...")

param_grid = {
    "C": [
        0.5,
        1.0,
        2.0
    ]
}


# ============================================================
# 15. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)


grid_search.fit(
    X_train,
    y_train_encoded
)


# ============================================================
# 16. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 60)

print(
    "\nBest Parameters:"
)

print(
    grid_search.best_params_
)

print(
    f"\nBest Cross-Validation Accuracy: "
    f"{grid_search.best_score_ * 100:.2f}%"
)


# ============================================================
# 17. TEST BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

test_accuracy = best_model.score(
    X_test,
    y_test_encoded
)

print(
    f"\nBest Model Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 18. SAVE BEST MODEL
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

import joblib

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_regression_tuned.pkl"
)

joblib.dump(
    best_model,
    MODEL_PATH
)


# ============================================================
# 19. SAVE TUNING RESULTS
# ============================================================

RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "hyperparameter_tuning_results.txt"
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Logistic Regression Hyperparameter Tuning\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Baseline Test Accuracy: "
        f"{baseline_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Best Parameters: "
        f"{grid_search.best_params_}\n"
    )

    file.write(
        f"Best CV Accuracy: "
        f"{grid_search.best_score_ * 100:.2f}%\n"
    )

    file.write(
        f"Best Model Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("TUNING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nTuned model saved to:"
)

print(
    MODEL_PATH
)

print(
    "\nResults saved to:"
)

print(
    RESULTS_PATH
)

print("\nDone.")