"""
Job Description Matching
AI-Powered Career Intelligence Platform
"""

import re


# ============================================================
# KNOWN SKILLS
# ============================================================

KNOWN_SKILLS = {
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "angular",
    "node.js",
    "node",
    "flask",
    "django",
    "fastapi",

    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",

    "git",
    "github",
    "docker",
    "kubernetes",

    "aws",
    "azure",
    "gcp",
    "cloud",

    "machine learning",
    "deep learning",
    "artificial intelligence",
    "ai",
    "nlp",
    "tensorflow",
    "pytorch",
    "scikit-learn",

    "numpy",
    "pandas",
    "matplotlib",
    "power bi",
    "tableau",
    "excel",
    "statistics",

    "data analysis",
    "data analytics",

    "rest api",
    "apis",
    "api",

    "testing",
    "selenium",

    "communication",
    "leadership",
    "problem solving",
    "teamwork",

    "agile",
    "scrum"
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Convert text into normalized lowercase format.
    """

    text = str(text or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EXTRACT SKILLS FROM JOB DESCRIPTION
# ============================================================

def extract_job_skills(job_description):
    """
    Extract recognized skills from the job description.
    """

    text = normalize_text(
        job_description
    )

    found_skills = []

    for skill in KNOWN_SKILLS:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
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


# ============================================================
# EXTRACT SKILLS FROM RESUME
# ============================================================

def extract_resume_skills(resume_details):
    """
    Extract recognized skills from resume_details.

    This function handles both:

    1. Skills stored as a list
    2. Skills stored as a string

    Only skills from KNOWN_SKILLS are returned.
    """

    skills = resume_details.get(
        "skills",
        ""
    )

    # --------------------------------------------------------
    # Convert skills into one text value
    # --------------------------------------------------------

    if isinstance(
        skills,
        list
    ):

        full_skill_text = " ".join(
            str(skill)
            for skill in skills
        )

    else:

        full_skill_text = str(
            skills or ""
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized_full_text = normalize_text(
        full_skill_text
    )

    resume_skills = set()

    # --------------------------------------------------------
    # Detect recognized skills
    # --------------------------------------------------------

    for skill in KNOWN_SKILLS:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            normalized_full_text
        ):

            resume_skills.add(
                skill
            )

    return resume_skills


# ============================================================
# CALCULATE JOB MATCH
# ============================================================

def calculate_job_match(
    resume_details,
    job_description
):
    """
    Compare resume skills with job-description skills.

    Returns:

        match_score
        job_skills
        matching_skills
        missing_skills
        resume_skills
        recommendations
    """

    job_description = job_description or ""

    # --------------------------------------------------------
    # EMPTY JOB DESCRIPTION
    # --------------------------------------------------------

    if not job_description.strip():

        return {

            "match_score": 0,

            "job_skills": [],

            "matching_skills": [],

            "missing_skills": [],

            "resume_skills": [],

            "recommendations": [
                "Paste a job description to calculate the match."
            ]

        }

    # --------------------------------------------------------
    # EXTRACT JOB SKILLS
    # --------------------------------------------------------

    job_skills = set(
        extract_job_skills(
            job_description
        )
    )

    # --------------------------------------------------------
    # EXTRACT RESUME SKILLS
    # --------------------------------------------------------

    resume_skills = extract_resume_skills(
        resume_details
    )

    # --------------------------------------------------------
    # FIND MATCHING SKILLS
    # --------------------------------------------------------

    matching_skills = sorted(
        job_skills.intersection(
            resume_skills
        )
    )

    # --------------------------------------------------------
    # FIND MISSING SKILLS
    # --------------------------------------------------------

    missing_skills = sorted(
        job_skills.difference(
            resume_skills
        )
    )

    # --------------------------------------------------------
    # CALCULATE MATCH SCORE
    # --------------------------------------------------------

    if len(job_skills) > 0:

        match_score = round(
            (
                len(matching_skills)
                /
                len(job_skills)
            )
            * 100,
            1
        )

    else:

        match_score = 0

    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = []

    if missing_skills:

        for skill in missing_skills[:8]:

            recommendations.append(
                f"Add or develop {skill} "
                f"to improve your job match."
            )

    else:

        recommendations.append(
            "Your resume contains the main "
            "skills detected in this job description."
        )

    # --------------------------------------------------------
    # NO RECOGNIZED SKILLS
    # --------------------------------------------------------

    if not job_skills:

        recommendations = [

            "The system could not identify "
            "standard skills in this job description.",

            "Try a job description containing "
            "specific technologies, tools, "
            "or professional skills."

        ]

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "match_score": match_score,

        "job_skills": sorted(
            job_skills
        ),

        "matching_skills": matching_skills,

        "missing_skills": missing_skills,

        "resume_skills": sorted(
            resume_skills
        ),

        "recommendations": recommendations

    }