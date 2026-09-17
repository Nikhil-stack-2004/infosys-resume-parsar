"""
Skill Gap Analysis
AI-Powered Career Intelligence Platform
"""

import re


# ============================================================
# CAREER SKILL REQUIREMENTS
# ============================================================

CAREER_SKILLS = {

    "Technology": [
        "python",
        "java",
        "sql",
        "git",
        "apis",
        "cloud",
        "testing"
    ],

    "Data & Analytics": [
        "python",
        "sql",
        "numpy",
        "pandas",
        "power bi",
        "statistics",
        "tableau"
    ],

    "Consulting": [
        "communication",
        "problem solving",
        "excel",
        "sql",
        "data analysis",
        "presentation",
        "leadership"
    ]
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# NORMALIZE RESUME SKILLS
# ============================================================

def normalize_resume_skills(skills):

    if skills is None:
        return set()

    # --------------------------------------------------------
    # List format
    # --------------------------------------------------------

    if isinstance(skills, list):

        text = " ".join(
            normalize_text(skill)
            for skill in skills
        )

    # --------------------------------------------------------
    # String format
    # --------------------------------------------------------

    else:

        text = normalize_text(skills)

    # Replace separators

    text = text.replace(",", " ")
    text = text.replace("|", " ")
    text = text.replace(";", " ")
    text = text.replace("/", " ")

    words = set(
        text.split()
    )

    # --------------------------------------------------------
    # Also keep complete multi-word skills
    # --------------------------------------------------------

    for skill in [
        "machine learning",
        "deep learning",
        "data analysis",
        "data analytics",
        "power bi",
        "problem solving",
        "artificial intelligence",
        "rest api"
    ]:

        if skill in text:

            words.add(skill)

    return words


# ============================================================
# CHECK WHETHER SKILL EXISTS
# ============================================================

def skill_exists(
    resume_skills,
    required_skill
):

    required_skill = normalize_text(
        required_skill
    )

    # Exact match

    if required_skill in resume_skills:
        return True

    # Check inside complete resume skill text

    resume_text = " ".join(
        resume_skills
    )

    if required_skill in resume_text:
        return True

    # Aliases

    aliases = {

        "apis": [
            "api",
            "apis",
            "rest api",
            "rest apis"
        ],

        "cloud": [
            "cloud",
            "aws",
            "azure",
            "gcp"
        ],

        "git": [
            "git",
            "github"
        ],

        "sql": [
            "sql",
            "mysql",
            "postgresql",
            "oracle"
        ],

        "testing": [
            "testing",
            "selenium"
        ],

        "python": [
            "python"
        ],

        "java": [
            "java"
        ],

        "excel": [
            "excel"
        ],

        "communication": [
            "communication"
        ],

        "problem solving": [
            "problem solving",
            "problem-solving"
        ],

        "data analysis": [
            "data analysis",
            "data analytics"
        ]
    }

    for alias in aliases.get(
        required_skill,
        []
    ):

        if alias in resume_skills:

            return True

        if alias in resume_text:

            return True

    return False


# ============================================================
# CALCULATE ONE CAREER GAP
# ============================================================

def calculate_career_gap(
    resume_skills,
    career
):

    required_skills = CAREER_SKILLS.get(
        career,
        []
    )

    matching_skills = []
    missing_skills = []

    for skill in required_skills:

        if skill_exists(
            resume_skills,
            skill
        ):

            matching_skills.append(
                skill
            )

        else:

            missing_skills.append(
                skill
            )

    total = len(
        required_skills
    )

    if total > 0:

        readiness = round(
            (
                len(matching_skills)
                / total
            ) * 100,
            1
        )

    else:

        readiness = 0.0

    return {

        "career": career,

        "readiness": readiness,

        "matching_skills": matching_skills,

        "missing_skills": missing_skills,

        "recommendations": [
            f"Build a small {career} project "
            f"that demonstrates {skill}."
            for skill in missing_skills[:8]
        ]
    }


# ============================================================
# GET CAREER NAME FROM PREDICTION
# ============================================================

def get_prediction_name(
    prediction
):

    # --------------------------------------------------------
    # Dictionary prediction
    # --------------------------------------------------------

    if isinstance(
        prediction,
        dict
    ):

        possible_keys = [
            "role",
            "job_role",
            "career",
            "label",
            "name",
            "title",
            "class"
        ]

        for key in possible_keys:

            value = prediction.get(
                key
            )

            if value:

                return str(
                    value
                ).strip()

    # --------------------------------------------------------
    # String prediction
    # --------------------------------------------------------

    if isinstance(
        prediction,
        str
    ):

        return prediction.strip()

    return None


# ============================================================
# NORMALIZE CAREER NAME
# ============================================================

def normalize_career_name(
    career
):

    if not career:
        return None

    text = normalize_text(
        career
    )

    # Technology

    if (
        "technology" in text
        or "tech" in text
        or "software" in text
        or "developer" in text
        or "engineering" in text
    ):

        return "Technology"

    # Data

    if (
        "data" in text
        or "analytics" in text
        or "analyst" in text
    ):

        return "Data & Analytics"

    # Consulting

    if "consult" in text:

        return "Consulting"

    return career.strip()


# ============================================================
# BUILD GAP REPORTS
# ============================================================

def build_gap_reports(
    skills,
    predictions
):

    resume_skills = normalize_resume_skills(
        skills
    )

    careers = []

    # --------------------------------------------------------
    # Extract predicted careers
    # --------------------------------------------------------

    if isinstance(
        predictions,
        list
    ):

        for prediction in predictions:

            career = get_prediction_name(
                prediction
            )

            career = normalize_career_name(
                career
            )

            if career:

                careers.append(
                    career
                )

    elif isinstance(
        predictions,
        dict
    ):

        # Case 1:
        # {"Technology": 89.8, ...}

        for key in predictions.keys():

            career = normalize_career_name(
                key
            )

            if career:

                careers.append(
                    career
                )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_careers = []

    for career in careers:

        if career not in unique_careers:

            unique_careers.append(
                career
            )

    # --------------------------------------------------------
    # IMPORTANT:
    # If predictor returned unexpected format,
    # use the three platform careers.
    # --------------------------------------------------------

    if not unique_careers:

        unique_careers = [
            "Technology",
            "Data & Analytics",
            "Consulting"
        ]

    # --------------------------------------------------------
    # Generate reports
    # --------------------------------------------------------

    reports = []

    for career in unique_careers:

        result = calculate_career_gap(
            resume_skills,
            career
        )

        missing_skills = result[
            "missing_skills"
        ]

        recommendations = result[
            "recommendations"
        ]

        # ----------------------------------------------------
        # No missing skills
        # ----------------------------------------------------

        if not missing_skills:

            recommendations = [
                "Your current resume skills "
                f"cover the main skills required "
                f"for {career}."
            ]

        reports.append({

            "career": career,

            "readiness": result[
                "readiness"
            ],

            "matching_skills": result[
                "matching_skills"
            ],

            "missing_skills": missing_skills,

            "recommendations": recommendations
        })

    return reports