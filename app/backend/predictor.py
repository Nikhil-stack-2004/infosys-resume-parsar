import os
import joblib
import numpy as np
from scipy.sparse import hstack


# ============================================================
# LOAD MODELS
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


model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "logistic_regression_model.pkl"
    )
)

tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)

education_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "education_onehot_encoder.pkl"
    )
)

label_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "job_role_label_encoder.pkl"
    )
)


# ============================================================
# JOB ROLE PREDICTION
# ============================================================

def predict_job_roles(
    resume_text,
    education,
    experience,
    skills,
    top_n=3
):
    """
    Predict the most relevant job roles for a resume.
    """

    resume_text = str(resume_text or "")
    skills = str(skills or "")
    education = str(education or "").lower()

    try:
        experience = float(experience or 0)
    except (ValueError, TypeError):
        experience = 0.0


    # --------------------------------------------------------
    # Combine resume text and skills
    # --------------------------------------------------------

    combined_text = (
        resume_text
        + " "
        + skills
    )


    # --------------------------------------------------------
    # TF-IDF features
    # --------------------------------------------------------

    X_text = tfidf.transform(
        [combined_text]
    )


    # --------------------------------------------------------
    # Education features
    # --------------------------------------------------------

    education_vector = education_encoder.transform(
        [[education]]
    )


    # --------------------------------------------------------
    # Experience feature
    # --------------------------------------------------------

    experience_vector = np.array(
        [[experience]]
    )


    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    X_final = hstack(
        [
            X_text,
            experience_vector,
            education_vector
        ]
    )


    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_final
    )[0]


    top_indices = np.argsort(
        probabilities
    )[::-1][:top_n]


    predicted_labels = model.classes_[
        top_indices
    ]


    job_roles = label_encoder.inverse_transform(
        predicted_labels
    )


    top_probabilities = probabilities[
        top_indices
    ]


    probability_total = top_probabilities.sum()


    # --------------------------------------------------------
    # Relative matching
    # --------------------------------------------------------

    if probability_total > 0:

        relative_matches = (
            top_probabilities
            /
            probability_total
        )

    else:

        relative_matches = np.zeros(
            len(top_indices)
        )


    # --------------------------------------------------------
    # User-friendly match percentages
    # --------------------------------------------------------

    match_percentages = []


    for rank, relative_match in enumerate(
        relative_matches
    ):

        if rank == 0:

            score = (
                70
                +
                (relative_match * 20)
            )

        elif rank == 1:

            score = (
                60
                +
                (relative_match * 20)
            )

        else:

            score = (
                40
                +
                (relative_match * 20)
            )


        match_percentages.append(
            score
        )


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return [
        {
            "role": role.title(),
            "confidence": round(
                float(match_percentage),
                1
            )
        }

        for role, match_percentage
        in zip(
            job_roles,
            match_percentages
        )
    ]