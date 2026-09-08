"""FastAPI REST service for career prediction and skill-gap reports."""

from typing import List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from app.backend.gap_analysis import build_gap_report, build_gap_reports
    from app.backend.predictor import predict_job_roles
except ModuleNotFoundError:
    from backend.gap_analysis import build_gap_report, build_gap_reports
    from backend.predictor import predict_job_roles

app = FastAPI(title="Career Intelligence API", version="3.0.0")


class ResumeRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    education: str = "high school"
    experience: Union[int, float] = Field(default=0, ge=0)
    skills: Union[str, List[str]] = ""
    top_n: int = Field(default=3, ge=1, le=3)


def _skills_text(skills):
    return skills if isinstance(skills, str) else " ".join(skills)


def _predict(request):
    return predict_job_roles(
        request.resume_text,
        request.education,
        request.experience,
        _skills_text(request.skills),
        request.top_n,
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "career-intelligence"}


@app.post("/predict")
def predict(request: ResumeRequest):
    return {"predictions": _predict(request)}


@app.post("/recommendations")
def recommendations(request: ResumeRequest):
    predictions = _predict(request)
    return {
        "predictions": predictions,
        "recommendations": [
            {
                "career_area": prediction["role"],
                "match": prediction["confidence"],
                "next_step": f"Close the highest-priority skill gaps for {prediction['role']}",
            }
            for prediction in predictions
        ],
    }


@app.post("/gap-report")
def gap_report(request: ResumeRequest):
    predictions = _predict(request)
    return {
        "predictions": predictions,
        "reports": build_gap_reports(request.skills, predictions),
    }


@app.post("/gap-report/{career_area}")
def career_gap_report(career_area: str, request: ResumeRequest):
    predictions = _predict(request)
    known_areas = {prediction["role"] for prediction in predictions}
    if career_area not in known_areas:
        raise HTTPException(status_code=404, detail="Career area is not in the recommendations")
    return build_gap_report(request.skills, career_area)
