"""
Skill Gap Analysis
AI-Powered Career Intelligence Platform
"""

import os
import re
import json


# ============================================================
# PATHS
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

SKILL_PROFILE_FILE = os.path.join(
    MODEL_DIR,
    "job_role_skill_profiles.json"
)


# ============================================================
# FALLBACK CAREER SKILLS
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
# SKILL ALIASES
# ============================================================

# IMPORTANT:
# Aliases are deliberately conservative.
# We do NOT use broad substring matching.

SKILL_ALIASES = {

    "github": [
        "github",
        "git"
    ],

    "git": [
        "git",
        "github"
    ],

    "mysql": [
        "mysql",
        "sql"
    ],

    "postgresql": [
        "postgresql",
        "postgres",
        "sql"
    ],

    "postgres": [
        "postgres",
        "postgresql",
        "sql"
    ],

    "oracle": [
        "oracle",
        "sql"
    ],

    "sql": [
        "sql",
        "mysql",
        "postgres",
        "postgresql",
        "oracle"
    ],

    "api": [
        "api",
        "apis",
        "rest api",
        "rest apis"
    ],

    "apis": [
        "api",
        "apis",
        "rest api",
        "rest apis"
    ],

    "rest api": [
        "rest api",
        "rest apis",
        "api",
        "apis"
    ],

    "rest apis": [
        "rest apis",
        "rest api",
        "api",
        "apis"
    ],

    "machine learning": [
        "machine learning",
        "ml"
    ],

    "ml": [
        "ml",
        "machine learning"
    ],

    "deep learning": [
        "deep learning",
        "dl"
    ],

    "dl": [
        "dl",
        "deep learning"
    ],

    "data analysis": [
        "data analysis",
        "data analytics"
    ],

    "data analytics": [
        "data analytics",
        "data analysis"
    ],

    "problem solving": [
        "problem solving",
        "problem-solving",
        "problem solving skills"
    ],

    "communication": [
        "communication",
        "communication skills"
    ],

    "excel": [
        "excel",
        "microsoft excel"
    ],

    "microsoft excel": [
        "microsoft excel",
        "excel"
    ],

    "cloud": [
        "cloud"
    ],

    "aws": [
        "aws"
    ],

    "azure": [
        "azure"
    ],

    "gcp": [
        "gcp"
    ],

    "testing": [
        "testing",
        "software testing",
        "selenium"
    ],

    "software testing": [
        "software testing",
        "testing",
        "selenium"
    ],

    "selenium": [
        "selenium",
        "testing"
    ],

    "python": [
        "python"
    ],

    "java": [
        "java"
    ],

    "c": [
        "c",
        "c programming",
        "c language"
    ],

    "csharp": [
        "csharp",
        "c#"
    ],

    "cpp": [
        "cpp",
        "c++"
    ],

    "c++": [
        "c++",
        "cpp"
    ]
}


# ============================================================
# LOAD TRAINED JOB ROLE SKILL PROFILES
# ============================================================

def load_skill_profiles():

    if not os.path.exists(
        SKILL_PROFILE_FILE
    ):
        print(
            "Warning: job_role_skill_profiles.json not found."
        )
        return {}

    try:

        with open(
            SKILL_PROFILE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, dict):

            print(
                f"Loaded {len(data)} job-role skill profiles."
            )

            return data

    except Exception as error:

        print(
            f"Warning: Could not load skill profiles: {error}"
        )

    return {}


JOB_ROLE_SKILL_PROFILES = load_skill_profiles()


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = value.replace(
        "_",
        " "
    )

    value = value.replace(
        "-",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# NORMALIZE SKILL
# ============================================================

def normalize_skill(skill):

    skill = normalize_text(
        skill
    )

    replacements = {
        "c plus plus": "cpp",
        "c sharp": "csharp",
        "c #": "csharp",
        "microsoft excel": "excel"
    }

    if skill in replacements:
        skill = replacements[skill]

    skill = re.sub(
        r"\s+",
        " ",
        skill
    )

    return skill.strip()


# ============================================================
# SPLIT SKILL STRING
# ============================================================

def split_skill_string(text):

    text = normalize_text(
        text
    )

    if not text:
        return []

    # Multi-word skills must be detected first.
    multi_word_skills = [
        "machine learning",
        "deep learning",
        "data analysis",
        "data analytics",
        "power bi",
        "problem solving",
        "artificial intelligence",
        "rest api",
        "rest apis",
        "computer vision",
        "natural language processing",
        "database management",
        "web development",
        "software development",
        "software testing",
        "user interface",
        "user experience",
        "technical knowledge",
        "attention to detail",
        "mobile development",
        "android development",
        "business acumen",
        "business development",
        "java interop",
        "penetration testing",
        "scikit learn",
        "mobile ui"
    ]

    found = []

    # Sort longest first.
    multi_word_skills.sort(
        key=len,
        reverse=True
    )

    for skill in multi_word_skills:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            text
        ):

            found.append(
                skill
            )

            text = re.sub(
                pattern,
                " ",
                text
            )

    # Remaining single-word skills.
    text = re.sub(
        r"[,|;/]+",
        " ",
        text
    )

    found.extend(
        text.split()
    )

    return found


# ============================================================
# NORMALIZE RESUME SKILLS
# ============================================================

def normalize_resume_skills(skills):

    if skills is None:
        return set()

    # --------------------------------------------------------
    # Convert input into list
    # --------------------------------------------------------

    if isinstance(
        skills,
        (list, tuple, set)
    ):

        values = list(
            skills
        )

    else:

        values = split_skill_string(
            skills
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = set()

    for skill in values:

        skill = normalize_skill(
            skill
        )

        if skill:
            normalized.add(
                skill
            )

    return normalized


# ============================================================
# EXTRACT SKILLS FROM ROLE PROFILE
# ============================================================

def extract_profile_skills(profile):

    if isinstance(
        profile,
        list
    ):
        return profile

    if isinstance(
        profile,
        tuple
    ):
        return list(profile)

    if isinstance(
        profile,
        set
    ):
        return list(profile)

    if isinstance(
        profile,
        str
    ):
        return [
            profile
        ]

    if isinstance(
        profile,
        dict
    ):

        possible_keys = [
            "skills",
            "required_skills",
            "skill",
            "skill_profile",
            "top_skills"
        ]

        for key in possible_keys:

            if key in profile:

                value = profile[key]

                if isinstance(
                    value,
                    dict
                ):
                    return list(
                        value.keys()
                    )

                if isinstance(
                    value,
                    list
                ):
                    return value

                if isinstance(
                    value,
                    str
                ):
                    return [
                        value
                    ]

        return list(
            profile.keys()
        )

    return []


# ============================================================
# FIND SKILLS FOR PREDICTED ROLE
# ============================================================

def get_role_skills(career):

    if not career:
        return []

    career_text = normalize_text(
        career
    )

    # --------------------------------------------------------
    # Exact role match
    # --------------------------------------------------------

    for role, profile in JOB_ROLE_SKILL_PROFILES.items():

        if normalize_text(
            role
        ) == career_text:

            skills = extract_profile_skills(
                profile
            )

            if skills:
                return skills

    # --------------------------------------------------------
    # Partial role match
    # --------------------------------------------------------

    for role, profile in JOB_ROLE_SKILL_PROFILES.items():

        role_text = normalize_text(
            role
        )

        if (
            career_text in role_text
            or role_text in career_text
        ):

            skills = extract_profile_skills(
                profile
            )

            if skills:
                return skills

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    normalized_career = normalize_career_name(
        career
    )

    return CAREER_SKILLS.get(
        normalized_career,
        []
    )


# ============================================================
# GET ACCEPTED SKILLS
# ============================================================

def get_accepted_skill_set(required_skill):

    required_skill = normalize_skill(
        required_skill
    )

    accepted = {
        required_skill
    }

    aliases = SKILL_ALIASES.get(
        required_skill,
        []
    )

    for alias in aliases:

        alias = normalize_skill(
            alias
        )

        if alias:
            accepted.add(
                alias
            )

    return accepted


# ============================================================
# CHECK WHETHER SKILL EXISTS
# ============================================================

def skill_exists(
    resume_skills,
    required_skill
):
    """
    Strict skill matching.

    Allowed:
        Exact normalized match
        Explicit alias match

    Not allowed:
        Broad substring matching

    This prevents:
        machine learning -> cnc machines
        machine learning -> market research
        machine learning -> technical knowledge
    """

    required_skill = normalize_skill(
        required_skill
    )

    if not required_skill:
        return False

    # Exact match
    if required_skill in resume_skills:
        return True

    # Explicit aliases only
    accepted_skills = get_accepted_skill_set(
        required_skill
    )

    return bool(
        accepted_skills.intersection(
            resume_skills
        )
    )


# ============================================================
# CALCULATE ONE CAREER GAP
# ============================================================

def calculate_career_gap(
    resume_skills,
    career
):

    # Normalize resume skills.
    resume_skills = normalize_resume_skills(
        resume_skills
    )

    # Get role-specific skills.
    required_skills = get_role_skills(
        career
    )

    # Normalize required skills.
    normalized_required = []

    for skill in required_skills:

        skill = normalize_skill(
            skill
        )

        if (
            skill
            and skill not in normalized_required
        ):

            normalized_required.append(
                skill
            )

    # Compare.
    matching_skills = []
    missing_skills = []

    for skill in normalized_required:

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

    # --------------------------------------------------------
    # Readiness
    # --------------------------------------------------------

    total = len(
        normalized_required
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

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = []

    for skill in missing_skills[:8]:

        recommendations.append(
            f"Develop {skill} skills through "
            f"a practical {career} project."
        )

    if not recommendations:

        recommendations = [
            "Your current resume skills cover "
            f"the main skills identified for {career}."
        ]

    return {
        "career": career,
        "readiness": readiness,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "recommendations": recommendations
    }


# ============================================================
# GET CAREER NAME FROM PREDICTION
# ============================================================

def get_prediction_name(
    prediction
):

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

    if text in [
        "technology",
        "tech"
    ]:

        return "Technology"

    if text in [
        "data",
        "data analytics",
        "data & analytics",
        "analytics"
    ]:

        return "Data & Analytics"

    if text in [
        "consulting",
        "consultant"
    ]:

        return "Consulting"

    return str(
        career
    ).strip()


# ============================================================
# BUILD GAP REPORTS
# ============================================================

def build_gap_reports(
    skills,
    predictions
):

    # Normalize resume skills once.
    resume_skills = normalize_resume_skills(
        skills
    )

    print(
        "Normalized resume skills:",
        sorted(resume_skills)
    )

    careers = []

    # --------------------------------------------------------
    # Prediction list
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

    # --------------------------------------------------------
    # Prediction dictionary
    # --------------------------------------------------------

    elif isinstance(
        predictions,
        dict
    ):

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
    # Fallback
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

        reports.append(
            {
                "career": result["career"],
                "readiness": result["readiness"],
                "matching_skills": result["matching_skills"],
                "missing_skills": result["missing_skills"],
                "recommendations": result["recommendations"]
            }
        )

    return reports