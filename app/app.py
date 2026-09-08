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

# Create Flask application
app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "career-intelligence-development-key")

# -----------------------------
# Load Machine Learning Models
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "job_role_label_encoder.pkl"))
education_encoder = joblib.load(os.path.join(MODEL_DIR, "education_onehot_encoder.pkl"))

print("✅ All models loaded successfully!")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# -----------------------------
# Home Route
# -----------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if request.method == "GET":
        return render_template("upload.html")

    if "resume" not in request.files:
        flash("Please choose a PDF resume to upload.", "error")
        return redirect(url_for("upload_page"))

    file = request.files["resume"]
    filename = secure_filename(file.filename or "")
    if not filename:
        flash("Please choose a PDF resume to upload.", "error")
        return redirect(url_for("upload_page"))
    if not filename.lower().endswith(".pdf"):
        flash("Only PDF resumes are supported right now.", "error")
        return redirect(url_for("upload_page"))

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        resume_text = extract_text_from_pdf(filepath)
        education = extract_education(resume_text)
        experience = extract_experience_years(resume_text)
        skills = extract_skills(resume_text)
        email = extract_email(resume_text)
        phone = extract_phone(resume_text)
        resume_details = {
            "name": resume_text.split("\n")[0].strip(),
            "email": email,
            "education": education,
            "skills": skills,
            "phone": phone,
            "experience": f"{experience} Years" if experience > 0 else "Fresher",
        }
        predictions = predict_job_roles(resume_text, education, experience, skills)
        gap_reports = build_gap_reports(skills, predictions)
    except Exception:
        flash("We could not read that PDF. Please try another resume.", "error")
        return redirect(url_for("upload_page"))

    session["analysis"] = {
        "resume_details": resume_details,
        "predictions": predictions,
        "gap_reports": gap_reports,
        "filename": filename,
    }
    return redirect(url_for("prediction"))


@app.route("/prediction")
def prediction():
    return render_template("prediction.html", analysis=session.get("analysis"))

# -----------------------------
# Run Flask App
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
