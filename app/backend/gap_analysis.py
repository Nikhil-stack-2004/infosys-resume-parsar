"""Skill-gap analysis for predicted career areas."""

CAREER_SKILLS = {
    "Technology": {"python", "sql", "git", "cloud", "apis", "testing"},
    "Data & Analytics": {"python", "sql", "pandas", "numpy", "statistics", "tableau", "power bi"},
    "Engineering & Manufacturing": {"cad", "quality", "manufacturing", "safety", "project management"},
    "Finance & Accounting": {"excel", "accounting", "sql", "financial analysis", "auditing"},
    "Marketing & Sales": {"marketing", "sales", "seo", "analytics", "communication", "crm"},
    "Creative & Design": {"design", "figma", "ux", "ui", "photoshop", "communication"},
    "Human Resources": {"recruiting", "communication", "hr", "employee relations", "analytics"},
    "Healthcare": {"patient care", "medical", "healthcare", "documentation", "communication"},
    "Education & Training": {"teaching", "training", "communication", "curriculum", "presentation"},
    "Operations & Supply Chain": {"logistics", "procurement", "excel", "inventory", "project management"},
}

SKILL_ALIASES = {
    "power bi": "power bi",
    "powerbi": "power bi",
    "machine learning": "python",
    "deep learning": "python",
    "rest api": "apis",
    "rest apis": "apis",
    "javascript": "apis",
    "mysql": "sql",
}


def _normalise_skills(skills):
    if isinstance(skills, str):
        values = {item.strip().lower() for item in skills.split(",") if item.strip()}
    else:
        values = {str(item).strip().lower() for item in (skills or []) if str(item).strip()}
    return {SKILL_ALIASES.get(value, value) for value in values}


def build_gap_report(skills, career_area):
    """Return matched skills, gaps, and actionable learning suggestions."""
    target_skills = CAREER_SKILLS.get(career_area, set())
    known_skills = _normalise_skills(skills)
    matched = sorted(target_skills & known_skills)
    gaps = sorted(target_skills - known_skills)
    suggestions = [
        {
            "skill": skill,
            "action": f"Build a small {career_area} project that demonstrates {skill}.",
            "priority": "high" if index < 2 else "medium",
        }
        for index, skill in enumerate(gaps)
    ]
    score = round((len(matched) / len(target_skills)) * 100, 1) if target_skills else 0.0
    return {
        "career_area": career_area,
        "readiness": score,
        "matched_skills": matched,
        "skill_gaps": gaps,
        "suggestions": suggestions,
    }


def build_gap_reports(skills, predictions):
    return [build_gap_report(skills, prediction["role"]) for prediction in predictions]
