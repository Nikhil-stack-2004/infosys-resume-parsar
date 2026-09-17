import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack


# ============================================================
# LOAD PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD TRAINED MODEL ARTIFACTS
# ============================================================

print("Loading career prediction models...")

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "logistic_regression_model.pkl"
    )
)

resume_tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)

skills_tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "skills_tfidf.pkl"
    )
)

education_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "education_onehot_encoder.pkl"
    )
)

job_role_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "job_role_label_encoder.pkl"
    )
)


# ============================================================
# MODEL CONSISTENCY CHECK
# ============================================================

EXPECTED_FEATURES = 8258
EXPECTED_CLASSES = 324

actual_features = getattr(
    model,
    "n_features_in_",
    None
)

actual_classes = len(
    getattr(
        model,
        "classes_",
        []
    )
)

encoder_classes = len(
    job_role_encoder.classes_
)

if actual_features != EXPECTED_FEATURES:
    raise ValueError(
        f"Model feature mismatch: expected "
        f"{EXPECTED_FEATURES}, found {actual_features}"
    )

if actual_classes != EXPECTED_CLASSES:
    raise ValueError(
        f"Model class mismatch: expected "
        f"{EXPECTED_CLASSES}, found {actual_classes}"
    )

if encoder_classes != EXPECTED_CLASSES:
    raise ValueError(
        f"Label encoder mismatch: expected "
        f"{EXPECTED_CLASSES}, found {encoder_classes}"
    )

print(
    f"Model verified: {actual_features} features, "
    f"{actual_classes} classes"
)

print("Career prediction models loaded successfully!")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean text using the same preprocessing logic
    used during model training.
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Keep letters, numbers and spaces
    text = "".join(
        char
        if char.isalnum() or char.isspace()
        else " "
        for char in text
    )

    # Remove duplicate spaces
    text = " ".join(
        text.split()
    )

    return text.strip()


# ============================================================
# EDUCATION PREPARATION
# ============================================================

def prepare_education(education):
    """
    Prepare education value for the saved OneHotEncoder.

    The encoder was trained with string values. Unknown
    education values are safely handled by handle_unknown='ignore'.
    """

    if education is None:
        return "unknown"

    education = str(
        education
    ).strip()

    if not education:
        return "unknown"

    return education


# ============================================================
# EXPERIENCE PREPARATION
# ============================================================

def prepare_experience(experience):
    """
    Convert experience to a numeric value.
    Invalid or missing values become 0.
    """

    try:
        value = float(
            experience
        )

        if value < 0:
            return 0.0

        return value

    except (
        ValueError,
        TypeError
    ):
        return 0.0


# ============================================================
# PREDICT JOB ROLES
# ============================================================

def predict_job_roles(
    resume_text,
    education,
    experience,
    skills,
    top_n=3
):
    """
    Predict the top job roles for a resume.

    Feature order MUST exactly match training:

        1. Resume TF-IDF
        2. Skills TF-IDF
        3. Education One-Hot
        4. Experience
    """

    # --------------------------------------------------------
    # PREPARE INPUT
    # --------------------------------------------------------

    resume_text = clean_text(
        resume_text
    )

    skills = clean_text(
        skills
    )

    education = prepare_education(
        education
    )

    experience = prepare_experience(
        experience
    )


    # --------------------------------------------------------
    # RESUME TF-IDF
    # --------------------------------------------------------

    X_resume = resume_tfidf.transform(
        [resume_text]
    )


    # --------------------------------------------------------
    # SKILLS TF-IDF
    # --------------------------------------------------------

    X_skills = skills_tfidf.transform(
        [skills]
    )


    # --------------------------------------------------------
    # EDUCATION ONE-HOT
    # --------------------------------------------------------

    education_input = pd.DataFrame(
        {
            "Education": [education]
        }
    )

    X_education = education_encoder.transform(
        education_input
    )


    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    X_experience = np.array(
        [
            [experience]
        ],
        dtype=float
    )


    # --------------------------------------------------------
    # COMBINE FEATURES
    # --------------------------------------------------------
    #
    # EXACT TRAINING ORDER:
    #
    # Resume TF-IDF = 5000
    # Skills TF-IDF = 3000
    # Education = 257
    # Experience = 1
    #
    # TOTAL = 8258
    #

    X_final = hstack(
        [
            X_resume,
            X_skills,
            X_education,
            X_experience
        ]
    ).tocsr()


    # --------------------------------------------------------
    # FEATURE VALIDATION
    # --------------------------------------------------------

    actual_features = X_final.shape[1]

    expected_features = getattr(
        model,
        "n_features_in_",
        None
    )

    if (
        expected_features is not None
        and actual_features != expected_features
    ):
        raise ValueError(
            "Feature mismatch: "
            f"model expects {expected_features} "
            f"features, but predictor generated "
            f"{actual_features} features."
        )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_final
    )[0]


    # --------------------------------------------------------
    # TOP N ROLES
    # --------------------------------------------------------

    try:
        top_n = int(top_n)
    except (
        ValueError,
        TypeError
    ):
        top_n = 3

    top_n = max(
        1,
        min(
            top_n,
            len(probabilities)
        )
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:top_n]


    # --------------------------------------------------------
    # DECODE JOB ROLES
    # --------------------------------------------------------
    #
    # model.classes_ contains encoded class IDs.
    # job_role_encoder converts those IDs back to role names.
    #

    predicted_class_ids = model.classes_[
        top_indices
    ]

    job_roles = job_role_encoder.inverse_transform(
        predicted_class_ids
    )


    # --------------------------------------------------------
    # BUILD RESULTS
    # --------------------------------------------------------

    results = []

    for role, index in zip(
        job_roles,
        top_indices
    ):

        probability = float(
            probabilities[index]
        )

        results.append(
            {
                "role": str(
                    role
                ),

                "confidence": round(
                    probability * 100,
                    2
                )
            }
        )


    return results