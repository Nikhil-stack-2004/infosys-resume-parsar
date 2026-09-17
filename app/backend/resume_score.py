"""Resume quality scoring for the AI Career Intelligence Platform."""

import re


def calculate_resume_score(resume_text, resume_details):
    """
    Calculate a transparent 0-100 resume quality score.

    The score is based on resume completeness rather than judging
    the person's qualifications.
    """

    text = (resume_text or "").lower()

    score = 0
    breakdown = []

    # Contact information - 15 points
    contact_score = 0

    if resume_details.get("email"):
        contact_score += 8

    if resume_details.get("phone"):
        contact_score += 7

    score += contact_score
    breakdown.append({
        "category": "Contact Information",
        "score": contact_score,
        "max_score": 15
    })

    # Education - 15 points
    education_score = 15 if resume_details.get("education") else 0
    score += education_score

    breakdown.append({
        "category": "Education",
        "score": education_score,
        "max_score": 15
    })

    # Skills - 20 points
    skills = resume_details.get("skills", "")
    skill_list = [s for s in skills.split() if s.strip()]

    if len(skill_list) >= 8:
        skill_score = 20
    elif len(skill_list) >= 5:
        skill_score = 15
    elif len(skill_list) >= 3:
        skill_score = 10
    elif len(skill_list) >= 1:
        skill_score = 5
    else:
        skill_score = 0

    score += skill_score

    breakdown.append({
        "category": "Technical Skills",
        "score": skill_score,
        "max_score": 20
    })

    # Projects - 15 points
    project_keywords = [
        "project",
        "projects",
        "developed",
        "built",
        "implemented",
        "created"
    ]

    project_found = any(keyword in text for keyword in project_keywords)

    project_score = 15 if project_found else 0
    score += project_score

    breakdown.append({
        "category": "Projects",
        "score": project_score,
        "max_score": 15
    })

    # Experience / internship - 15 points
    experience_keywords = [
        "experience",
        "internship",
        "intern",
        "work experience",
        "professional experience"
    ]

    experience_found = any(keyword in text for keyword in experience_keywords)

    if resume_details.get("experience") != "Fresher" or experience_found:
        experience_score = 15
    else:
        # A fresher is not penalized heavily.
        experience_score = 8

    score += experience_score

    breakdown.append({
        "category": "Experience",
        "score": experience_score,
        "max_score": 15
    })

    # Certifications - 10 points
    certification_keywords = [
        "certification",
        "certifications",
        "certificate",
        "certified",
        "course",
        "courses"
    ]

    certification_found = any(
        keyword in text for keyword in certification_keywords
    )

    certification_score = 10 if certification_found else 0
    score += certification_score

    breakdown.append({
        "category": "Certifications",
        "score": certification_score,
        "max_score": 10
    })

    # Resume structure - 10 points
    sections = [
        "education",
        "skills",
        "projects",
        "experience",
        "objective",
        "summary"
    ]

    sections_found = sum(1 for section in sections if section in text)

    if sections_found >= 5:
        structure_score = 10
    elif sections_found >= 3:
        structure_score = 7
    elif sections_found >= 2:
        structure_score = 4
    else:
        structure_score = 0

    score += structure_score

    breakdown.append({
        "category": "Resume Structure",
        "score": structure_score,
        "max_score": 10
    })

    score = min(100, round(score))

    # Improvement suggestions
    suggestions = []

    if not resume_details.get("phone"):
        suggestions.append("Add a professional phone number.")

    if not project_found:
        suggestions.append("Add 2-3 relevant academic or personal projects.")

    if not certification_found:
        suggestions.append("Add relevant certifications or completed courses.")

    if len(skill_list) < 5:
        suggestions.append("Add more relevant technical skills.")

    if sections_found < 4:
        suggestions.append("Improve resume structure with clear sections.")

    return {
        "score": score,
        "breakdown": breakdown,
        "suggestions": suggestions
    }