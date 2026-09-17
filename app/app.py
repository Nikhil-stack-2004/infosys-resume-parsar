from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
    jsonify
)

import joblib
import os
import traceback

from werkzeug.utils import secure_filename


# ============================================================
# BACKEND IMPORTS
# ============================================================

from backend.parser import extract_text_from_pdf

from backend.feature_extractor import (
    extract_education,
    extract_experience_years,
    extract_skills,
    extract_email,
    extract_phone
)

from backend.predictor import predict_job_roles

from backend.gap_analysis import (
    build_gap_reports
)

from backend.resume_score import (
    calculate_resume_score
)

from backend.job_matcher import (
    calculate_job_match
)

from backend.interview_coach import (
    generate_interview_questions,
    evaluate_answer
)

from backend.routes import (
    extract_skills as api_extract_skills
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
# BASE / MODEL DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD MACHINE LEARNING MODELS
# ============================================================

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

    print()
    print("=" * 70)
    print("❌ MODEL LOADING ERROR")
    print("=" * 70)
    print("Error type:", type(error).__name__)
    print("Error:", repr(error))
    traceback.print_exc()
    print("=" * 70)

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
# HELPER - NORMALIZE SKILLS
# ============================================================

def normalize_skills(skills):
    """
    Convert skills into a clean list.

    Supports:
    - list
    - tuple
    - set
    - comma-separated string
    - space-separated string
    """

    if skills is None:
        return []

    # --------------------------------------------------------
    # Already a list
    # --------------------------------------------------------

    if isinstance(skills, list):

        result = skills

    # --------------------------------------------------------
    # Tuple / set
    # --------------------------------------------------------

    elif isinstance(skills, (tuple, set)):

        result = list(skills)

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    elif isinstance(skills, str):

        text = skills.strip()

        if not text:
            return []

        # Prefer comma-separated format
        if "," in text:

            result = [
                item.strip()
                for item in text.split(",")
                if item.strip()
            ]

        else:

            # A simple space-separated skills string
            # is handled as individual tokens.
            result = text.split()

    else:

        result = []

    # --------------------------------------------------------
    # Clean values
    # --------------------------------------------------------

    cleaned = []

    for skill in result:

        if skill is None:
            continue

        value = str(skill).strip()

        if value:
            cleaned.append(value)

    return sorted(
        set(cleaned),
        key=lambda value: value.lower()
    )


# ============================================================
# PDF TEXT EXTRACTION HELPER
# ============================================================

def read_resume_pdf(filepath):

    """
    Extract text from a PDF.

    Method 1:
        Existing project parser

    Method 2:
        pypdf

    Method 3:
        PyPDF2
    """

    errors = []

    # --------------------------------------------------------
    # METHOD 1 - PROJECT PARSER
    # --------------------------------------------------------

    try:

        print("📖 Trying project PDF parser...")

        text = extract_text_from_pdf(
            filepath
        )

        if text:

            text = str(text).strip()

            if text:

                print(
                    "✅ Project PDF parser succeeded."
                )

                return text

        errors.append(
            "Project parser returned empty text."
        )

    except Exception as error:

        errors.append(
            "Project parser: "
            f"{type(error).__name__}: {error}"
        )

        print(
            "⚠️ Project parser failed:",
            repr(error)
        )


    # --------------------------------------------------------
    # METHOD 2 - PYPDF
    # --------------------------------------------------------

    try:

        print("📖 Trying pypdf fallback...")

        from pypdf import PdfReader

        reader = PdfReader(
            filepath
        )

        pages = []

        for page in reader.pages:

            try:

                page_text = page.extract_text()

                if page_text:

                    pages.append(
                        str(page_text)
                    )

            except Exception as page_error:

                print(
                    "⚠️ Could not read PDF page:",
                    repr(page_error)
                )

        text = "\n".join(
            pages
        ).strip()

        if text:

            print(
                "✅ pypdf fallback succeeded."
            )

            return text

        errors.append(
            "pypdf returned empty text."
        )

    except Exception as error:

        errors.append(
            "pypdf: "
            f"{type(error).__name__}: {error}"
        )

        print(
            "⚠️ pypdf fallback failed:",
            repr(error)
        )


    # --------------------------------------------------------
    # METHOD 3 - PYPDF2
    # --------------------------------------------------------

    try:

        print("📖 Trying PyPDF2 fallback...")

        from PyPDF2 import PdfReader

        reader = PdfReader(
            filepath
        )

        pages = []

        for page in reader.pages:

            try:

                page_text = page.extract_text()

                if page_text:

                    pages.append(
                        str(page_text)
                    )

            except Exception as page_error:

                print(
                    "⚠️ Could not read PDF page:",
                    repr(page_error)
                )

        text = "\n".join(
            pages
        ).strip()

        if text:

            print(
                "✅ PyPDF2 fallback succeeded."
            )

            return text

        errors.append(
            "PyPDF2 returned empty text."
        )

    except Exception as error:

        errors.append(
            "PyPDF2: "
            f"{type(error).__name__}: {error}"
        )

        print(
            "⚠️ PyPDF2 fallback failed:",
            repr(error)
        )


    # --------------------------------------------------------
    # ALL METHODS FAILED
    # --------------------------------------------------------

    raise ValueError(
        "Unable to extract readable text from PDF.\n"
        + "\n".join(errors)
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
    # GET
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "upload.html"
        )


    # --------------------------------------------------------
    # CHECK FILE FIELD
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

    if file is None:

        flash(
            "No resume file was received.",
            "error"
        )

        return redirect(
            url_for("upload_page")
        )


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
    # VALIDATE EXTENSION
    # --------------------------------------------------------

    if not filename.lower().endswith(".pdf"):

        flash(
            "Only PDF resumes are supported.",
            "error"
        )

        return redirect(
            url_for("upload_page")
        )


    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:

        file.save(
            filepath
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("❌ FILE SAVE ERROR")
        print("=" * 70)
        print("Error type:", type(error).__name__)
        print("Error:", repr(error))
        traceback.print_exc()
        print("=" * 70)

        flash(
            "The resume could not be saved.",
            "error"
        )

        return redirect(
            url_for("upload_page")
        )


    # --------------------------------------------------------
    # UPLOAD INFORMATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("📄 RESUME UPLOADED")
    print("=" * 70)

    print(
        "Filename:",
        filename
    )

    print(
        "Path:",
        filepath
    )

    print(
        "File exists:",
        os.path.exists(filepath)
    )

    if os.path.exists(filepath):

        print(
            "File size:",
            os.path.getsize(filepath),
            "bytes"
        )

    print("=" * 70)


    # ========================================================
    # PROCESS RESUME
    # ========================================================

    try:

        # ----------------------------------------------------
        # PDF TEXT
        # ----------------------------------------------------

        print(
            "📖 Reading PDF..."
        )

        resume_text = read_resume_pdf(
            filepath
        )

        if not resume_text:

            raise ValueError(
                "No readable text was extracted from the PDF."
            )

        resume_text = str(
            resume_text
        ).strip()

        if not resume_text:

            raise ValueError(
                "PDF contains no readable text."
            )


        print(
            "✅ Resume text extracted"
        )

        print(
            "Text length:",
            len(resume_text)
        )

        print(
            "Text preview:",
            resume_text[:300].replace(
                "\n",
                " "
            )
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

        extracted_skills = extract_skills(
            resume_text
        )

        skills = normalize_skills(
            extracted_skills
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
        # NAME
        # ====================================================

        first_line = ""

        for line in resume_text.splitlines():

            cleaned_line = line.strip()

            if cleaned_line:

                first_line = cleaned_line

                break


        # ====================================================
        # IMPORTANT COMPATIBILITY FIX
        # ====================================================

        # Keep skills as a STRING inside resume_details.
        #
        # Some existing modules such as resume_score.py
        # may expect resume_details["skills"] to be a string.
        #
        # Gap analysis still receives the real list.
        #

        skills_string = ", ".join(
            skills
        )


        # ====================================================
        # CREATE RESUME PROFILE
        # ====================================================

        resume_details = {

            "name": first_line,

            "email": email,

            "education": education,

            "skills": skills_string,

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

        print(
            "Profile skills:",
            resume_details["skills"]
        )


        # ====================================================
        # CAREER PREDICTION
        # ====================================================

        print(
            "🤖 Generating career predictions..."
        )


        # Predictor expects the extracted skills
        # as a compatible text representation.

        skills_text = " ".join(
            skills
        )


        predictions = predict_job_roles(

            resume_text,

            education,

            experience,

            skills_text
        )


        # ----------------------------------------------------
        # NORMALIZE PREDICTIONS
        # ----------------------------------------------------

        if predictions is None:

            predictions = []

        elif isinstance(
            predictions,
            tuple
        ):

            predictions = list(
                predictions
            )

        elif isinstance(
            predictions,
            dict
        ):

            predictions = []

        elif not isinstance(
            predictions,
            list
        ):

            predictions = []


        print(
            "🤖 Career predictions:",
            predictions
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


        # Make sure result is dictionary-compatible

        if resume_score is None:

            resume_score = {}

        elif not isinstance(
            resume_score,
            dict
        ):

            resume_score = {
                "score": resume_score
            }


        print(
            "⭐ Resume Quality Score:",
            resume_score.get(
                "score",
                0
            )
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        print()
        print("=" * 70)
        print("❌ RESUME PROCESSING ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(error).__name__
        )

        print(
            "Error:",
            str(error)
        )

        print(
            "Error repr:",
            repr(error)
        )

        print()
        print(
            "FULL TRACEBACK:"
        )

        traceback.print_exc()

        print("=" * 70)


        # ----------------------------------------------------
        # REMOVE FAILED FILE
        # ----------------------------------------------------

        try:

            if os.path.exists(filepath):

                os.remove(
                    filepath
                )

        except Exception as remove_error:

            print(
                "Could not remove failed file:",
                repr(remove_error)
            )


        # ----------------------------------------------------
        # SHOW REAL ERROR
        # ----------------------------------------------------

        flash(
            "Resume processing failed: "
            f"{type(error).__name__}: {str(error)}",
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


    # --------------------------------------------------------
    # CLEAR OLD JOB MATCH DATA
    # --------------------------------------------------------

    session.pop(
        "job_match",
        None
    )

    session.pop(
        "job_description",
        None
    )


    print()
    print("=" * 70)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()


    # ========================================================
    # REDIRECT
    # ========================================================

    return redirect(
        url_for("prediction")
    )


# ============================================================
# PREDICTION / CAREER ANALYSIS
# ============================================================

@app.route(
    "/prediction"
)
def prediction():

    analysis = session.get(
        "analysis"
    )


    if not analysis:

        flash(
            "Please upload a resume before viewing your career analysis.",
            "error"
        )

        return redirect(
            url_for("upload_page")
        )


    resume_details = analysis.get(
        "resume_details",
        {}
    ) or {}


    predictions = analysis.get(
        "predictions",
        []
    ) or []


    # --------------------------------------------------------
    # NORMALIZE PREDICTIONS
    # --------------------------------------------------------

    if isinstance(
        predictions,
        dict
    ):

        predictions = []

    elif isinstance(
        predictions,
        tuple
    ):

        predictions = list(
            predictions
        )

    elif not isinstance(
        predictions,
        list
    ):

        predictions = []


    resume_score = analysis.get(
        "resume_score",
        {}
    ) or {}


    skills = resume_details.get(
        "skills",
        []
    ) or []


    # --------------------------------------------------------
    # NORMALIZE SKILLS
    # --------------------------------------------------------

    skills = normalize_skills(
        skills
    )


    # --------------------------------------------------------
    # REBUILD GAP REPORTS
    # --------------------------------------------------------

    try:

        gap_reports = build_gap_reports(
            skills,
            predictions
        )

    except Exception as error:

        print(
            "⚠️ Gap report rebuild failed:",
            repr(error)
        )

        traceback.print_exc()

        gap_reports = []


    analysis["gap_reports"] = gap_reports

    session["analysis"] = analysis


    return render_template(

        "prediction.html",

        analysis=analysis,

        profile=resume_details,

        resume_details=resume_details,

        predictions=predictions,

        gap_reports=gap_reports,

        resume_score=resume_score,

        filename=analysis.get(
            "filename",
            ""
        )
    )


# ============================================================
# JOB DESCRIPTION MATCHING
# ============================================================

@app.route(
    "/job-match",
    methods=["GET", "POST"]
)
def job_match():

    analysis = session.get(
        "analysis"
    )


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
    ) or {}


    job_match_result = session.get(
        "job_match"
    )


    job_description = session.get(
        "job_description",
        ""
    )


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


        job_match_result = calculate_job_match(

            resume_details,

            job_description
        )


        session["job_match"] = (
            job_match_result
        )

        session["job_description"] = (
            job_description
        )


    return render_template(

        "job_match.html",

        analysis=analysis,

        resume_details=resume_details,

        job_match=job_match_result,

        job_description=job_description
    )


# ============================================================
# AI INTERVIEW COACH
# ============================================================

@app.route(
    "/interview-coach",
    methods=["GET", "POST"]
)
def interview_coach():

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


    resume_details = analysis.get(
        "resume_details",
        {}
    ) or {}


    candidate_name = resume_details.get(
        "name",
        "Candidate"
    )


    resume_skills = resume_details.get(
        "skills",
        []
    )


    skills_list = normalize_skills(
        resume_skills
    )


    predictions = analysis.get(
        "predictions",
        []
    ) or []


    if isinstance(
        predictions,
        list
    ) and predictions:

        first_prediction = predictions[0]

        if isinstance(
            first_prediction,
            dict
        ):

            target_role = first_prediction.get(
                "role",
                "Technology"
            )

        else:

            target_role = "Technology"

    else:

        target_role = "Technology"


    # --------------------------------------------------------
    # RESET
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
    # GENERATE QUESTIONS
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
    # CURRENT INDEX
    # --------------------------------------------------------

    current_index = session.get(
        "interview_current_index",
        0
    )


    # --------------------------------------------------------
    # SUBMIT ANSWER
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


        evaluation = evaluate_answer(

            current_question,

            answer,

            skills_list
        )


        interview_results = session.get(
            "interview_results",
            []
        )


        if not isinstance(
            interview_results,
            list
        ):

            interview_results = []


        interview_results.append({

            "question": current_question,

            "answer": answer,

            "evaluation": evaluation

        })


        session[
            "interview_results"
        ] = interview_results


        session[
            "interview_current_index"
        ] = current_index + 1


        return redirect(
            url_for(
                "interview_coach"
            )
        )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    interview_results = session.get(
        "interview_results",
        []
    )


    if not isinstance(
        interview_results,
        list
    ):

        interview_results = []


    interview_complete = (
        current_index >= len(
            questions
        )
    )


    final_score = 0


    if interview_results:

        scores = []

        for result in interview_results:

            evaluation = result.get(
                "evaluation"
            )

            if isinstance(
                evaluation,
                dict
            ):

                score = evaluation.get(
                    "overall_score"
                )

                if score is not None:

                    try:

                        scores.append(
                            float(score)
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        pass


        if scores:

            final_score = round(
                sum(scores) / len(scores)
            )


    current_question = None


    if not interview_complete:

        current_question = questions[
            current_index
        ]


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
# REST API - HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def api_health():

    return jsonify({

        "success": True,

        "status": "healthy",

        "service":
            "AI Career Intelligence Platform API"

    }), 200


# ============================================================
# REST API - SKILL EXTRACTION
# ============================================================

@app.route(
    "/api/extract-skills",
    methods=["POST"]
)
def api_extract_skills_route():

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({

            "success": False,

            "error":
                "JSON request body is required"

        }), 400


    text = str(
        data.get(
            "text",
            ""
        )
    ).strip()


    if not text:

        return jsonify({

            "success": False,

            "error": "text is required"

        }), 400


    skills = api_extract_skills(
        text
    )


    if skills is None:

        skills = []


    return jsonify({

        "success": True,

        "skills": skills,

        "count": len(skills)

    }), 200


# ============================================================
# REST API - JOB MATCH
# ============================================================

@app.route(
    "/api/job-match",
    methods=["POST"]
)
def api_job_match():

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({

            "success": False,

            "error":
                "JSON request body is required"

        }), 400


    job_description = str(
        data.get(
            "job_description",
            ""
        )
    ).strip()


    resume_skills = data.get(
        "resume_skills",
        []
    )


    # --------------------------------------------------------
    # VALIDATE JOB DESCRIPTION
    # --------------------------------------------------------

    if not job_description:

        return jsonify({

            "success": False,

            "error":
                "job_description is required"

        }), 400


    # --------------------------------------------------------
    # NORMALIZE RESUME SKILLS
    # --------------------------------------------------------

    if isinstance(
        resume_skills,
        str
    ):

        if "," in resume_skills:

            resume_skills = [

                skill.strip()

                for skill
                in resume_skills.split(",")

                if skill.strip()

            ]

        else:

            resume_skills = resume_skills.split()


    if not isinstance(
        resume_skills,
        list
    ):

        return jsonify({

            "success": False,

            "error":
                "resume_skills must be a list "
                "or comma-separated string"

        }), 400


    # --------------------------------------------------------
    # EXTRACT REQUIRED SKILLS
    # --------------------------------------------------------

    required_skills = api_extract_skills(
        job_description
    )


    if required_skills is None:

        required_skills = []


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    resume_skills_set = {

        str(skill).strip().lower()

        for skill
        in resume_skills

        if str(skill).strip()

    }


    required_skills_set = {

        str(skill).strip().lower()

        for skill
        in required_skills

        if str(skill).strip()

    }


    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    matching = sorted(

        resume_skills_set.intersection(
            required_skills_set
        )

    )


    # --------------------------------------------------------
    # MISSING
    # --------------------------------------------------------

    missing = sorted(

        required_skills_set
        -
        resume_skills_set

    )


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    if required_skills_set:

        score = round(

            (
                len(matching)
                /
                len(required_skills_set)
            )
            *
            100,

            1
        )

    else:

        score = 0


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return jsonify({

        "success": True,

        "match_score": score,

        "matching_skills": matching,

        "missing_skills": missing,

        "required_skills":
            sorted(
                required_skills_set
            )

    }), 200


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )