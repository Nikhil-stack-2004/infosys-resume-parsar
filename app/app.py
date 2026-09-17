from flask import Flask, flash, redirect, render_template, request, session, url_for
import joblib
import os
from werkzeug.utils import secure_filename

from backend.parser import extract_text_from_pdf

from backend.feature_extractor import (
    extract_education,
    extract_experience_years,
    extract_skills,
    extract_email,
    extract_phone
)

from backend.predictor import predict_job_roles
from backend.gap_analysis import build_gap_reports
from backend.resume_score import calculate_resume_score
from backend.job_matcher import calculate_job_match

from backend.interview_coach import (
    generate_interview_questions,
    evaluate_answer
)


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "career-intelligence-development-key"
)


# ============================================================
# LOAD MACHINE LEARNING MODELS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


try:

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

    label_encoder = joblib.load(
        os.path.join(
            MODEL_DIR,
            "job_role_label_encoder.pkl"
        )
    )

    education_encoder = joblib.load(
        os.path.join(
            MODEL_DIR,
            "education_onehot_encoder.pkl"
        )
    )

    print("✅ All models loaded successfully!")


except Exception as error:

    print("❌ Model loading error:", error)

    raise


# ============================================================
# UPLOAD CONFIGURATION
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# RESUME UPLOAD ROUTE
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload_page():

    # --------------------------------------------------------
    # SHOW UPLOAD PAGE
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "upload.html"
        )


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "resume" not in request.files:

        flash(
            "Please choose a PDF resume to upload.",
            "error"
        )

        return redirect(
            url_for("upload_page")
        )


    # --------------------------------------------------------
    # GET FILE
    # --------------------------------------------------------

    file = request.files["resume"]

    filename = secure_filename(
        file.filename or ""
    )


    # --------------------------------------------------------
    # VALIDATE FILENAME
    # --------------------------------------------------------

    if not filename:

        flash(
            "Please choose a PDF resume to upload.",
            "error"
        )

        return redirect(
            url_for("upload_page")
        )


    # --------------------------------------------------------
    # VALIDATE PDF
    # --------------------------------------------------------

    if not filename.lower().endswith(".pdf"):

        flash(
            "Only PDF resumes are supported right now.",
            "error"
        )

        return redirect(
            url_for("upload_page")
        )


    # --------------------------------------------------------
    # SAVE RESUME
    # --------------------------------------------------------

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(
        filepath
    )


    print()
    print("=" * 60)
    print("📄 RESUME UPLOADED")
    print("=" * 60)
    print("Filename:", filename)
    print("Path:", filepath)


    # ========================================================
    # PROCESS RESUME
    # ========================================================

    try:

        # ----------------------------------------------------
        # EXTRACT PDF TEXT
        # ----------------------------------------------------

        resume_text = extract_text_from_pdf(
            filepath
        )

        if not resume_text or not resume_text.strip():

            raise ValueError(
                "No readable text found in the PDF."
            )


        print(
            "✅ Resume text extracted"
        )


        # ----------------------------------------------------
        # EDUCATION
        # ----------------------------------------------------

        education = extract_education(
            resume_text
        )

        print(
            "🎓 Education:",
            education
        )


        # ----------------------------------------------------
        # EXPERIENCE
        # ----------------------------------------------------

        experience = extract_experience_years(
            resume_text
        )

        print(
            "💼 Experience:",
            experience
        )


        # ----------------------------------------------------
        # SKILLS
        # ----------------------------------------------------

        skills = extract_skills(
            resume_text
        )

        print(
            "🛠 Skills:",
            skills
        )


        # ----------------------------------------------------
        # EMAIL
        # ----------------------------------------------------

        email = extract_email(
            resume_text
        )

        print(
            "📧 Email:",
            email
        )


        # ----------------------------------------------------
        # PHONE
        # ----------------------------------------------------

        phone = extract_phone(
            resume_text
        )

        print(
            "📱 Phone:",
            phone
        )


        # ====================================================
        # CREATE RESUME PROFILE
        # ====================================================

        first_line = ""

        for line in resume_text.splitlines():

            cleaned_line = line.strip()

            if cleaned_line:

                first_line = cleaned_line

                break


        resume_details = {

            "name": first_line,

            "email": email,

            "education": education,

            "skills": skills,

            "phone": phone,

            "experience": (
                f"{experience} Years"
                if experience > 0
                else "Fresher"
            )
        }


        print(
            "✅ Resume profile created"
        )


        # ====================================================
        # CAREER PREDICTION
        # ====================================================

        predictions = predict_job_roles(
            resume_text,
            education,
            experience,
            skills
        )


        print(
            "🤖 Career predictions generated"
        )


        # ====================================================
        # SKILL GAP ANALYSIS
        # ====================================================

        gap_reports = build_gap_reports(
            skills,
            predictions
        )


        print(
            "📊 Skill gap analysis completed"
        )


        # ====================================================
        # RESUME QUALITY SCORE
        # ====================================================

        resume_score = calculate_resume_score(
            resume_text,
            resume_details
        )


        print(
            "⭐ Resume Quality Score:",
            resume_score.get(
                "score",
                0
            )
        )


    except Exception as error:

        print()
        print("=" * 60)
        print("❌ RESUME PROCESSING ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)


        flash(
            "We could not read that PDF. "
            "Please try another resume.",
            "error"
        )


        return redirect(
            url_for("upload_page")
        )


    # ========================================================
    # STORE COMPLETE ANALYSIS
    # ========================================================

    session["analysis"] = {

        "resume_details": resume_details,

        "predictions": predictions,

        "gap_reports": gap_reports,

        "resume_score": resume_score,

        "filename": filename
    }


    # Clear previous job matching results
    # whenever a new resume is uploaded.

    session.pop(
        "job_match",
        None
    )

    session.pop(
        "job_description",
        None
    )


    print()
    print("=" * 60)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()


    # ========================================================
    # REDIRECT TO PREDICTION PAGE
    # ========================================================

    return redirect(
        url_for("prediction")
    )


# ============================================================
# PREDICTION / ANALYSIS ROUTE
# ============================================================
@app.route("/prediction")
def prediction():

    # Get analysis stored after resume upload
    analysis = session.get("analysis")

    # If no resume has been uploaded
    if not analysis:
        flash(
            "Please upload a resume before viewing your career analysis.",
            "error"
        )

        return redirect(
            url_for("upload_page")
        )

    # Resume details
    resume_details = analysis.get(
        "resume_details",
        {}
    )

    # Career predictions
    predictions = analysis.get(
        "predictions",
        []
    )

    # Skill gap analysis
    gap_reports = analysis.get(
        "gap_reports",
        []
    )

    # Resume quality score
    resume_score = analysis.get(
        "resume_score",
        {}
    )

    return render_template(
        "prediction.html",

        resume_details=resume_details,

        predictions=predictions,

        gap_reports=gap_reports,

        resume_score=resume_score
    )

# ============================================================
# JOB DESCRIPTION MATCHING ROUTE
# ============================================================

@app.route(
    "/job-match",
    methods=["GET", "POST"]
)
def job_match():

    analysis = session.get("analysis")

    # Resume must be uploaded first
    if not analysis:
        flash(
            "Please upload a resume before using Job Match.",
            "error"
        )
        return redirect(
            url_for("upload_page")
        )

    resume_details = analysis.get(
        "resume_details",
        {}
    )

    job_match_result = session.get(
        "job_match"
    )

    job_description = session.get(
        "job_description",
        ""
    )

    # --------------------------------------------------------
    # PROCESS JOB DESCRIPTION
    # --------------------------------------------------------

    if request.method == "POST":

        job_description = request.form.get(
            "job_description",
            ""
        ).strip()

        if not job_description:

            flash(
                "Please paste a job description.",
                "error"
            )

            return redirect(
                url_for("job_match")
            )

        # Calculate match using existing job_matcher.py
        job_match_result = calculate_job_match(
            resume_details,
            job_description
        )

        # Store results in session
        session["job_match"] = job_match_result
        session["job_description"] = job_description

    return render_template(
        "job_match.html",
        analysis=analysis,
        resume_details=resume_details,
        job_match=job_match_result,
        job_description=job_description
    )

# ============================================================
# AI INTERVIEW COACH ROUTE
# ============================================================

@app.route(
    "/interview-coach",
    methods=["GET", "POST"]
)
def interview_coach():

    # --------------------------------------------------------
    # Get uploaded resume analysis
    # --------------------------------------------------------

    analysis = session.get(
        "analysis"
    )

    if not analysis:

        flash(
            "Please upload a resume before using Interview Coach.",
            "error"
        )

        return redirect(
            url_for("upload_page")
        )


    # --------------------------------------------------------
    # Resume details
    # --------------------------------------------------------

    resume_details = analysis.get(
        "resume_details",
        {}
    )


    candidate_name = resume_details.get(
        "name",
        "Candidate"
    )


    # --------------------------------------------------------
    # Resume skills
    # --------------------------------------------------------

    resume_skills = resume_details.get(
        "skills",
        ""
    )


    if isinstance(
        resume_skills,
        list
    ):

        skills_list = [
            str(skill).strip()
            for skill in resume_skills
            if str(skill).strip()
        ]

    else:

        skills_list = [
            skill.strip()
            for skill in str(
                resume_skills or ""
            ).split(",")
            if skill.strip()
        ]


    # --------------------------------------------------------
    # Predicted career role
    # --------------------------------------------------------

    predictions = analysis.get(
        "predictions",
        []
    )


    if predictions:

        target_role = predictions[0].get(
            "role",
            "Technology"
        )

    else:

        target_role = "Technology"


    # --------------------------------------------------------
    # Reset interview
    # --------------------------------------------------------

    if request.args.get(
        "reset"
    ) == "1":

        session.pop(
            "interview_questions",
            None
        )

        session.pop(
            "interview_current_index",
            None
        )

        session.pop(
            "interview_results",
            None
        )

        return redirect(
            url_for(
                "interview_coach"
            )
        )


    # --------------------------------------------------------
    # Generate questions
    # --------------------------------------------------------

    questions = session.get(
        "interview_questions"
    )


    if not questions:

        questions = generate_interview_questions(
            target_role,
            skills_list,
            5
        )

        session[
            "interview_questions"
        ] = questions


    # --------------------------------------------------------
    # Current question index
    # --------------------------------------------------------

    current_index = session.get(
        "interview_current_index",
        0
    )


    # --------------------------------------------------------
    # Submit answer
    # --------------------------------------------------------

    if request.method == "POST":

        answer = request.form.get(
            "answer",
            ""
        ).strip()


        if not answer:

            flash(
                "Please enter your answer before submitting.",
                "error"
            )

            return redirect(
                url_for(
                    "interview_coach"
                )
            )


        # Prevent invalid index

        if current_index >= len(
            questions
        ):

            return redirect(
                url_for(
                    "interview_coach"
                )
            )


        current_question = questions[
            current_index
        ]


        # ----------------------------------------------------
        # Evaluate answer
        # ----------------------------------------------------

        evaluation = evaluate_answer(
            current_question,
            answer,
            skills_list
        )


        # ----------------------------------------------------
        # Store interview result
        # ----------------------------------------------------

        interview_results = session.get(
            "interview_results",
            []
        )


        interview_results.append(
            {
                "question": current_question,

                "answer": answer,

                "evaluation": evaluation
            }
        )


        session[
            "interview_results"
        ] = interview_results


        # Move to next question

        session[
            "interview_current_index"
        ] = current_index + 1


        return redirect(
            url_for(
                "interview_coach"
            )
        )


    # --------------------------------------------------------
    # Get interview results
    # --------------------------------------------------------

    interview_results = session.get(
        "interview_results",
        []
    )


    # --------------------------------------------------------
    # Check completion
    # --------------------------------------------------------

    interview_complete = (
        current_index >= len(
            questions
        )
    )


    # --------------------------------------------------------
    # Calculate final score
    # --------------------------------------------------------

    final_score = 0


    if interview_results:

        scores = [
            result[
                "evaluation"
            ][
                "overall_score"
            ]

            for result
            in interview_results
        ]


        final_score = round(
            sum(scores)
            /
            len(scores)
        )


    # --------------------------------------------------------
    # Current question
    # --------------------------------------------------------

    current_question = None


    if not interview_complete:

        current_question = questions[
            current_index
        ]


    # --------------------------------------------------------
    # Render Interview Coach
    # --------------------------------------------------------

    return render_template(
        "interview.html",

        candidate_name=candidate_name,

        target_role=target_role,

        skills=skills_list,

        questions=questions,

        current_index=current_index,

        current_question=current_question,

        interview_results=interview_results,

        interview_complete=interview_complete,

        final_score=final_score
    )

# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
