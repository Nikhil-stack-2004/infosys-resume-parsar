import os
import re
import json
import pandas as pd
import numpy as np

from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clean_training_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# SKILL NORMALIZATION
# ============================================================

SKILL_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "angularjs": "angular",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "sql server": "sql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "git hub": "github",
    "restful api": "rest api",
    "restful apis": "rest api",
}


def normalize_skill(skill):
    skill = str(skill).strip().lower()

    skill = re.sub(
        r"[^a-z0-9+#.\s-]",
        "",
        skill
    )

    skill = re.sub(
        r"\s+",
        " ",
        skill
    ).strip()

    return SKILL_ALIASES.get(
        skill,
        skill
    )


# ============================================================
# EXTRACT SKILLS
# ============================================================

def extract_skills(text):
    """
    Extract skills from a comma/semicolon/pipe separated
    skill field.
    """

    if pd.isna(text):
        return set()

    text = str(text).lower()

    parts = re.split(
        r"[,;|/]+",
        text
    )

    skills = set()

    for part in parts:

        skill = normalize_skill(part)

        if skill and len(skill) >= 2:
            skills.add(skill)

    return skills


# ============================================================
# BUILD SKILL VOCABULARY
# ============================================================

def build_skill_vocabulary(df):

    counter = Counter()

    for value in df["Skills"]:

        skills = extract_skills(value)

        for skill in skills:
            counter[skill] += 1

    return counter


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("SKILL ALIGNMENT ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset records: {len(df)}"
)


required_columns = [
    "Resume Text",
    "Skills",
    "Job Role"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CREATE SKILL VOCABULARY
# ============================================================

print("\nBuilding skill vocabulary...")

skill_frequency = build_skill_vocabulary(
    df
)

print(
    f"Unique skills found: "
    f"{len(skill_frequency)}"
)


# ============================================================
# JOB ROLE → REQUIRED SKILLS
# ============================================================

print("\nBuilding job-role skill profiles...")

job_role_skills = {}

for role, group in df.groupby(
    "Job Role"
):

    role_counter = Counter()

    for skills_text in group["Skills"]:

        skills = extract_skills(
            skills_text
        )

        for skill in skills:
            role_counter[skill] += 1

    total_records = len(group)

    # Keep skills appearing in at least
    # 10% of the records for the role.
    required_skills = set()

    for skill, count in role_counter.items():

        frequency = count / total_records

        if frequency >= 0.10:
            required_skills.add(skill)

    # If the 10% threshold removes everything,
    # keep the most common skills.
    if not required_skills:

        required_skills = set(
            skill
            for skill, count
            in role_counter.most_common(10)
        )

    job_role_skills[
        str(role)
    ] = sorted(
        required_skills
    )


print(
    f"Job-role profiles created: "
    f"{len(job_role_skills)}"
)


# ============================================================
# SKILL ALIGNMENT FUNCTION
# ============================================================

def calculate_alignment(
    resume_skills,
    required_skills
):

    resume_skills = set(
        normalize_skill(skill)
        for skill in resume_skills
    )

    required_skills = set(
        normalize_skill(skill)
        for skill in required_skills
    )

    required_skills = {
        skill
        for skill in required_skills
        if skill
    }

    matched = (
        resume_skills
        & required_skills
    )

    missing = (
        required_skills
        - resume_skills
    )

    if len(required_skills) > 0:

        coverage = (
            len(matched)
            / len(required_skills)
        ) * 100

    else:

        coverage = 0.0

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "required_skill_count": len(
            required_skills
        ),
        "matched_skill_count": len(
            matched
        ),
        "alignment_percentage": round(
            coverage,
            2
        )
    }


# ============================================================
# CALCULATE ALIGNMENT FOR EVERY RESUME
# ============================================================

print("\nCalculating skill alignment...")

results = []

for index, row in df.iterrows():

    actual_role = str(
        row["Job Role"]
    )

    resume_skills = extract_skills(
        row["Skills"]
    )

    required_skills = job_role_skills.get(
        actual_role,
        []
    )

    alignment = calculate_alignment(
        resume_skills,
        required_skills
    )

    results.append(
        {
            "resume_index": index,
            "job_role": actual_role,
            "resume_skills": ", ".join(
                sorted(resume_skills)
            ),
            "required_skills": ", ".join(
                required_skills
            ),
            "matched_skills": ", ".join(
                alignment["matched_skills"]
            ),
            "missing_skills": ", ".join(
                alignment["missing_skills"]
            ),
            "required_skill_count":
                alignment[
                    "required_skill_count"
                ],
            "matched_skill_count":
                alignment[
                    "matched_skill_count"
                ],
            "skill_alignment_percentage":
                alignment[
                    "alignment_percentage"
                ]
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE ALIGNMENT RESULTS
# ============================================================

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "skill_alignment_results.csv"
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# OVERALL METRICS
# ============================================================

alignment_scores = results_df[
    "skill_alignment_percentage"
]

mean_alignment = (
    alignment_scores.mean()
)

median_alignment = (
    alignment_scores.median()
)

high_alignment = (
    alignment_scores >= 75
).mean() * 100

medium_alignment = (
    (
        alignment_scores >= 50
    )
    &
    (
        alignment_scores < 75
    )
).mean() * 100

low_alignment = (
    alignment_scores < 50
).mean() * 100


# ============================================================
# TOP SKILLS
# ============================================================

top_skills = (
    skill_frequency
    .most_common(20)
)


# ============================================================
# SAVE JOB ROLE SKILL PROFILES
# ============================================================

profiles_path = os.path.join(
    MODEL_DIR,
    "job_role_skill_profiles.json"
)

with open(
    profiles_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        job_role_skills,
        file,
        indent=4
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    MODEL_DIR,
    "skill_alignment_metrics.txt"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Skill Alignment Analysis\n"
    )

    file.write(
        "=" * 55 + "\n\n"
    )

    file.write(
        f"Dataset records: {len(df)}\n"
    )

    file.write(
        f"Unique skills: "
        f"{len(skill_frequency)}\n"
    )

    file.write(
        f"Job roles: "
        f"{len(job_role_skills)}\n\n"
    )

    file.write(
        f"Mean Skill Alignment: "
        f"{mean_alignment:.2f}%\n"
    )

    file.write(
        f"Median Skill Alignment: "
        f"{median_alignment:.2f}%\n"
    )

    file.write(
        f"High Alignment (>=75%): "
        f"{high_alignment:.2f}%\n"
    )

    file.write(
        f"Medium Alignment (50-74%): "
        f"{medium_alignment:.2f}%\n"
    )

    file.write(
        f"Low Alignment (<50%): "
        f"{low_alignment:.2f}%\n\n"
    )

    file.write(
        "Top 20 Skills:\n"
    )

    for skill, count in top_skills:

        file.write(
            f"{skill}: {count}\n"
        )


# ============================================================
# DISPLAY SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SKILL ALIGNMENT RESULTS")
print("=" * 70)

print(
    f"\nMean Skill Alignment: "
    f"{mean_alignment:.2f}%"
)

print(
    f"Median Skill Alignment: "
    f"{median_alignment:.2f}%"
)

print(
    f"High Alignment (>=75%): "
    f"{high_alignment:.2f}%"
)

print(
    f"Medium Alignment (50-74%): "
    f"{medium_alignment:.2f}%"
)

print(
    f"Low Alignment (<50%): "
    f"{low_alignment:.2f}%"
)


print(
    "\nTop 10 Skills:"
)

for skill, count in top_skills[:10]:

    print(
        f"  {skill}: {count}"
    )


print(
    "\nSample alignment results:"
)

print(
    results_df[
        [
            "job_role",
            "matched_skill_count",
            "required_skill_count",
            "skill_alignment_percentage"
        ]
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SKILL ALIGNMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nResults saved:"
)

print(
    RESULTS_PATH
)

print(
    "\nJob-role skill profiles saved:"
)

print(
    profiles_path
)

print(
    "\nMetrics saved:"
)

print(
    metrics_path
)

print("\nDone.")