import re
import os
import joblib


# -----------------------------------
# Load Education Encoder
# -----------------------------------

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

education_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "education_onehot_encoder.pkl"
    )
)

EDUCATION_CATEGORIES = education_encoder.categories_[0]


# -----------------------------------
# Education Extraction
# -----------------------------------

def extract_education(resume_text):

    text = resume_text.lower()

    # Computer Science
    if (
        "b.tech" in text
        or "btech" in text
        or "b.e" in text
        or "computer science" in text
    ):
        return "Bachelor's in Computer Science"

    # Information Technology
    elif (
        "information technology" in text
        or "bachelor of it" in text
        or "bachelor in it" in text
    ):
        return "Bachelor's in IT"

    # Business
    elif "mba" in text:
        return "MBA"

    elif "business administration" in text:
        return "Bachelor's in Business"

    # Mechanical
    elif "mechanical engineering" in text:
        return "Bachelor's in Mechanical Engineering"

    # Civil
    elif "civil engineering" in text:
        return "Bachelor's in Civil Engineering"

    # Electrical
    elif "electrical engineering" in text:
        return "Bachelor's in Electrical Engineering"

    # Electronics
    elif "electronics engineering" in text:
        return "Bachelor's in Electronics Engineering"

    # Artificial Intelligence
    elif (
        "artificial intelligence" in text
        or "aiml" in text
    ):
        return "Bachelor's in Computer Science"

    # Data Science
    elif "data science" in text:
        return "Master's in Data Science"

    # High School / Intermediate
    elif (
        "high school" in text
        or "intermediate" in text
    ):
        return "High School"

    return "High School"


# -----------------------------------
# Experience Extraction
# -----------------------------------

def extract_experience_years(resume_text):

    text = resume_text.lower()

    pattern = r"(\d+)\+?\s*years?"

    match = re.search(
        pattern,
        text
    )

    if match:
        return int(
            match.group(1)
        )

    return 0


# -----------------------------------
# Email Extraction
# -----------------------------------

def extract_email(resume_text):

    pattern = (
        r"[a-zA-Z0-9._%+-]+"
        r"@[a-zA-Z0-9.-]+"
        r"\.[a-zA-Z]{2,}"
    )

    match = re.search(
        pattern,
        resume_text
    )

    if match:
        return match.group()

    return ""


# -----------------------------------
# Phone Number Extraction
# -----------------------------------

def extract_phone(resume_text):

    pattern = r"(?:\+91[-\s]?)?[6-9]\d{9}"

    match = re.search(
        pattern,
        resume_text
    )

    if match:
        return match.group()

    return ""


# -----------------------------------
# Skills Extraction
# -----------------------------------

SKILLS = [

    "python",
    "java",
    "c",
    "c++",
    "html",
    "css",
    "javascript",
    "sql",
    "mysql",
    "mongodb",
    "react",
    "angular",
    "node.js",
    "flask",
    "django",
    "spring boot",
    "rest api",
    "machine learning",
    "deep learning",
    "tensorflow",
    "keras",
    "pandas",
    "numpy",
    "power bi",
    "excel",
    "git",
    "github",
    "docker",
    "aws"

]


def extract_skills(resume_text):

    if not resume_text:
        return []

    text = resume_text.lower()

    found_skills = []

    for skill in SKILLS:

        pattern = (
            r"\b"
            + re.escape(skill)
            + r"\b"
        )

        if re.search(
            pattern,
            text
        ):
            found_skills.append(
                skill
            )

    return sorted(
        set(found_skills)
    )