import re
from flask import render_template, request, jsonify


def extract_skills(text):
    """
    Extract common technical and professional skills
    from a job description.
    """

    skills = [
        "python",
        "java",
        "c",
        "c++",
        "javascript",
        "html",
        "css",
        "sql",
        "mysql",
        "mongodb",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "data science",
        "numpy",
        "pandas",
        "power bi",
        "tableau",
        "excel",
        "git",
        "github",
        "apis",
        "rest api",
        "flask",
        "django",
        "react",
        "node.js",
        "cloud",
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "testing",
        "communication",
        "leadership",
        "problem solving",
        "data analysis",
        "presentation"
    ]

    text = text.lower()
    found = []

    for skill in skills:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


def calculate_match(resume_skills, required_skills):
    """
    Calculate percentage match between resume skills
    and job-description skills.
    """

    resume_skills = {
        skill.strip().lower()
        for skill in resume_skills
    }

    required_skills = {
        skill.strip().lower()
        for skill in required_skills
    }

    if not required_skills:
        return 0, [], []

    matching = sorted(
        resume_skills.intersection(required_skills)
    )

    missing = sorted(
        required_skills - resume_skills
    )

    score = round(
        (len(matching) / len(required_skills)) * 100,
        1
    )

    return score, matching, missing


def register_routes(app):
    """
    Register application routes.
    """

    # ---------------------------------------------------------
    # JOB MATCH WEB PAGE
    # ---------------------------------------------------------

    @app.route("/job-match", methods=["GET", "POST"])
    def job_match():

        match_result = None

        if request.method == "POST":

            job_description = request.form.get(
                "job_description",
                ""
            ).strip()

            resume_skills_text = request.form.get(
                "resume_skills",
                ""
            ).strip()

            resume_skills = [
                skill.strip()
                for skill in resume_skills_text.split(",")
                if skill.strip()
            ]

            required_skills = extract_skills(
                job_description
            )

            score, matching, missing = calculate_match(
                resume_skills,
                required_skills
            )

            match_result = {
                "score": score,
                "matching": matching,
                "missing": missing,
                "required": required_skills
            }

        return render_template(
            "job_match.html",
            match_result=match_result
        )


    # ---------------------------------------------------------
    # REST API - JOB MATCH
    # ---------------------------------------------------------

    @app.route("/api/job-match", methods=["POST"])
    def api_job_match():

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "success": False,
                "error": "JSON request body is required"
            }), 400

        job_description = str(
            data.get("job_description", "")
        ).strip()

        resume_skills = data.get(
            "resume_skills",
            []
        )

        if not job_description:
            return jsonify({
                "success": False,
                "error": "job_description is required"
            }), 400

        if isinstance(resume_skills, str):
            resume_skills = [
                skill.strip()
                for skill in resume_skills.split(",")
                if skill.strip()
            ]

        if not isinstance(resume_skills, list):
            return jsonify({
                "success": False,
                "error": "resume_skills must be a list or comma-separated string"
            }), 400

        required_skills = extract_skills(
            job_description
        )

        score, matching, missing = calculate_match(
            resume_skills,
            required_skills
        )

        return jsonify({
            "success": True,
            "match_score": score,
            "matching_skills": matching,
            "missing_skills": missing,
            "required_skills": required_skills
        }), 200


    # ---------------------------------------------------------
    # REST API - SKILL EXTRACTION
    # ---------------------------------------------------------

    @app.route("/api/extract-skills", methods=["POST"])
    def api_extract_skills():

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "success": False,
                "error": "JSON request body is required"
            }), 400

        text = str(
            data.get("text", "")
        ).strip()

        if not text:
            return jsonify({
                "success": False,
                "error": "text is required"
            }), 400

        skills = extract_skills(text)

        return jsonify({
            "success": True,
            "skills": skills,
            "count": len(skills)
        }), 200


    # ---------------------------------------------------------
    # API HEALTH CHECK
    # ---------------------------------------------------------

    @app.route("/api/health", methods=["GET"])
    def api_health():

        return jsonify({
            "success": True,
            "status": "healthy",
            "service": "AI Career Intelligence Platform API"
        }), 200