import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)
print("Columns:", list(data.columns))


# ============================================================
# HANDLE COLUMN NAMES
# ============================================================

# Support both the original notebook names and processed names.

column_map = {}

for col in data.columns:
    normalized = col.lower().strip()

    if normalized in ["resume text", "resume_text"]:
        column_map[col] = "Resume Text"

    elif normalized in ["skills", "skill"]:
        column_map[col] = "Skills"

    elif normalized in ["education"]:
        column_map[col] = "Education"

    elif normalized in [
        "experience years",
        "experience_years",
        "experience"
    ]:
        column_map[col] = "Experience Years"

    elif normalized in ["job role", "job_role", "role"]:
        column_map[col] = "Job Role"

data = data.rename(columns=column_map)

required_columns = [
    "Resume Text",
    "Skills",
    "Education",
    "Experience Years",
    "Job Role"
]

missing = [
    col for col in required_columns
    if col not in data.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN TEXT
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
# FEATURES / TARGET
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

print("\nSplitting dataset...")

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

print("\nCreating Resume TF-IDF...")

resume_tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_resume = resume_tfidf.fit_transform(
    X_train["Resume Text"]
)

X_test_resume = resume_tfidf.transform(
    X_test["Resume Text"]
)

print(
    "Resume features:",
    X_train_resume.shape
)


# ============================================================
# SKILLS TF-IDF
# ============================================================

print("\nCreating Skills TF-IDF...")

skills_tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_skills = skills_tfidf.fit_transform(
    X_train["Skills"]
)

X_test_skills = skills_tfidf.transform(
    X_test["Skills"]
)

print(
    "Skills features:",
    X_train_skills.shape
)


# ============================================================
# EDUCATION ENCODING
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

print(
    "Education features:",
    X_train_education.shape
)


# ============================================================
# EXPERIENCE
# ============================================================

X_train_experience = (
    X_train[["Experience Years"]].values
)

X_test_experience = (
    X_test[["Experience Years"]].values
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print("\nCombining features...")

X_train_final = hstack([
    X_train_resume,
    X_train_skills,
    X_train_education,
    X_train_experience
]).tocsr()

X_test_final = hstack([
    X_test_resume,
    X_test_skills,
    X_test_education,
    X_test_experience
]).tocsr()

print(
    "Final training shape:",
    X_train_final.shape
)

print(
    "Final testing shape:",
    X_test_final.shape
)


# ============================================================
# JOB ROLE ENCODING
# ============================================================

print("\nEncoding job roles...")

job_role_encoder = LabelEncoder()

y_train_encoded = job_role_encoder.fit_transform(
    y_train
)

y_test_encoded = job_role_encoder.transform(
    y_test
)

print(
    "Number of job roles:",
    len(job_role_encoder.classes_)
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_final,
    y_train_encoded
)


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating Random Forest...")

rf_pred = rf_model.predict(
    X_test_final
)

rf_accuracy = accuracy_score(
    y_test_encoded,
    rf_pred
)

print(
    "\nRandom Forest Accuracy:",
    rf_accuracy
)

print(
    "Random Forest Accuracy (%):",
    round(rf_accuracy * 100, 2)
)


# ============================================================
# SAVE MODEL + REQUIRED PREPROCESSORS
# ============================================================

print("\nSaving model files...")

joblib.dump(
    rf_model,
    os.path.join(
        MODEL_DIR,
        "random_forest_model.pkl"
    )
)

joblib.dump(
    resume_tfidf,
    os.path.join(
        MODEL_DIR,
        "resume_tfidf.pkl"
    )
)

joblib.dump(
    skills_tfidf,
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
    job_role_encoder,
    os.path.join(
        MODEL_DIR,
        "job_role_label_encoder.pkl"
    )
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\nSaved files:")

files = [
    "random_forest_model.pkl",
    "resume_tfidf.pkl",
    "skills_tfidf.pkl",
    "education_onehot_encoder.pkl",
    "job_role_label_encoder.pkl"
]

for filename in files:

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    print(
        filename,
        "→",
        os.path.exists(path)
    )

print("\nRandom Forest training completed successfully!")