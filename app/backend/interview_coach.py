"""
AI Interview Coach
AI-Powered Career Intelligence Platform
"""

import re


# ============================================================
# INTERVIEW QUESTION BANK
# ============================================================

QUESTION_BANK = {

    "software developer": [
        "Explain your experience with Python and how you have used it in a project.",
        "What is the difference between a list, tuple, and dictionary in Python?",
        "Explain object-oriented programming and its main principles.",
        "How would you design a REST API for a simple application?",
        "What is SQL and why is it useful in software development?",
        "Explain how you would debug a program that is producing incorrect results."
    ],

    "technology": [
        "Explain your strongest technical skill and how you have applied it.",
        "Describe a technical project that you have worked on.",
        "How do you approach solving a difficult programming problem?",
        "What is the role of APIs in modern software applications?",
        "How do you test and debug your applications?",
        "Which technology would you like to learn next and why?"
    ],

    "data & analytics": [
        "Explain how you would analyze a new dataset.",
        "What is the difference between data analysis and machine learning?",
        "How have you used Python for data analysis?",
        "What are pandas and NumPy used for?",
        "Explain the importance of data visualization.",
        "How would you handle missing values in a dataset?"
    ],

    "consulting": [
        "Tell me about a challenging problem you solved.",
        "How would you approach a problem with incomplete information?",
        "Describe a project where you worked as part of a team.",
        "How do you communicate a technical idea to a non-technical person?",
        "How would you analyze a business problem?",
        "Describe a situation where you demonstrated leadership."
    ]
}


# ============================================================
# NORMALIZE ROLE
# ============================================================

def normalize_role(role):
    role = str(role or "").lower().strip()

    if "software" in role:
        return "software developer"

    if "technology" in role:
        return "technology"

    if "data" in role or "analytics" in role:
        return "data & analytics"

    if "consult" in role:
        return "consulting"

    return "technology"


# ============================================================
# GENERATE QUESTIONS
# ============================================================

def generate_interview_questions(
    role,
    skills=None,
    number_of_questions=5
):
    """
    Generate interview questions based on
    predicted job role and resume skills.
    """

    normalized_role = normalize_role(role)

    questions = QUESTION_BANK.get(
        normalized_role,
        QUESTION_BANK["technology"]
    ).copy()


    # --------------------------------------------------------
    # Add skill-specific questions
    # --------------------------------------------------------

    skills_text = ""

    if isinstance(skills, list):

        skills_text = " ".join(
            str(skill)
            for skill in skills
        )

    else:

        skills_text = str(
            skills or ""
        )


    skills_text = skills_text.lower()


    if "python" in skills_text:

        questions.insert(
            0,
            "How have you used Python in your projects?"
        )


    if "machine learning" in skills_text:

        questions.insert(
            1,
            "Explain a machine learning project you have worked on."
        )


    if "java" in skills_text:

        questions.insert(
            1,
            "What Java concepts are you most comfortable working with?"
        )


    if "sql" in skills_text:

        questions.insert(
            1,
            "How would you use SQL to retrieve and analyze data?"
        )


    # Remove duplicates

    unique_questions = []

    for question in questions:

        if question not in unique_questions:

            unique_questions.append(
                question
            )


    return unique_questions[
        :number_of_questions
    ]


# ============================================================
# ANSWER EVALUATION
# ============================================================

def evaluate_answer(
    question,
    answer,
    resume_skills=None
):
    """
    Rule-based interview answer evaluation.

    This provides a local evaluation without
    requiring an external AI API.
    """

    question = str(
        question or ""
    ).strip()

    answer = str(
        answer or ""
    ).strip()


    skills = resume_skills or []


    if isinstance(skills, str):

        skills = [
            skill.strip()
            for skill in skills.split(",")
            if skill.strip()
        ]


    answer_lower = answer.lower()


    # --------------------------------------------------------
    # Basic answer length
    # --------------------------------------------------------

    word_count = len(
        re.findall(
            r"\b\w+\b",
            answer
        )
    )


    if word_count == 0:

        return {
            "technical_relevance": 0,
            "clarity": 0,
            "completeness": 0,
            "communication": 0,
            "overall_score": 0,
            "feedback": [
                "Please provide an answer before submitting."
            ]
        }


    # --------------------------------------------------------
    # Clarity
    # --------------------------------------------------------

    if word_count >= 80:
        clarity = 90

    elif word_count >= 50:
        clarity = 82

    elif word_count >= 30:
        clarity = 74

    elif word_count >= 15:
        clarity = 62

    else:
        clarity = 45


    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    completeness = min(
        95,
        45 + int(
            min(word_count, 100)
            * 0.5
        )
    )


    # --------------------------------------------------------
    # Technical relevance
    # --------------------------------------------------------

    matched_skills = []

    for skill in skills:

        skill_text = str(
            skill
        ).lower().strip()

        if (
            skill_text
            and skill_text in answer_lower
        ):

            matched_skills.append(
                skill
            )


    if matched_skills:

        technical_relevance = min(
            95,
            65
            +
            (
                len(matched_skills)
                * 10
            )
        )

    else:

        technical_relevance = 55


    # --------------------------------------------------------
    # Communication
    # --------------------------------------------------------

    communication_words = [
        "because",
        "therefore",
        "for example",
        "project",
        "experience",
        "result",
        "implemented",
        "developed",
        "solved"
    ]


    communication_matches = sum(
        1
        for word in communication_words
        if word in answer_lower
    )


    communication = min(
        95,
        60
        +
        (
            communication_matches
            * 5
        )
    )


    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    overall_score = round(
        (
            technical_relevance
            +
            clarity
            +
            completeness
            +
            communication
        )
        /
        4
    )


    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    feedback = []


    if word_count < 30:

        feedback.append(
            "Give a more detailed answer with a practical example."
        )


    if not matched_skills and skills:

        feedback.append(
            "Connect your answer to the technical skills listed in your resume."
        )


    if (
        "example" not in answer_lower
        and word_count >= 20
    ):

        feedback.append(
            "Add a specific example from your project or academic experience."
        )


    if (
        "project" not in answer_lower
        and word_count >= 20
    ):

        feedback.append(
            "Mention how you applied the concept in a project."
        )


    if not feedback:

        feedback.append(
            "Good response. Continue supporting your explanation with practical examples."
        )


    return {
        "technical_relevance": technical_relevance,
        "clarity": clarity,
        "completeness": completeness,
        "communication": communication,
        "overall_score": overall_score,
        "feedback": feedback,
        "matched_skills": matched_skills,
        "word_count": word_count
    }