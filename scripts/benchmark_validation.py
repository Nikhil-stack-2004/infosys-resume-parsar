import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# FILE PATHS
# ============================================================

SEMANTIC_PATH = os.path.join(
    DATA_DIR,
    "semantic_job_recommendations.csv"
)

SKILL_PATH = os.path.join(
    DATA_DIR,
    "skill_alignment_results.csv"
)

TRAINING_PATH = os.path.join(
    DATA_DIR,
    "clean_training_data.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CAREER INTELLIGENCE BENCHMARK VALIDATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking validation files...")

required_files = [
    SEMANTIC_PATH,
    SKILL_PATH,
    TRAINING_PATH
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    print(
        f"FOUND: {os.path.basename(path)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading evaluation data...")

semantic_df = pd.read_csv(
    SEMANTIC_PATH
)

skill_df = pd.read_csv(
    SKILL_PATH
)

training_df = pd.read_csv(
    TRAINING_PATH
)


print(
    f"Training records: {len(training_df)}"
)

print(
    f"Semantic evaluation records: "
    f"{len(semantic_df)}"
)

print(
    f"Skill alignment records: "
    f"{len(skill_df)}"
)


# ============================================================
# SEMANTIC TOP-K VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("SEMANTIC TOP-K VALIDATION")
print("-" * 70)

required_semantic_columns = [
    "resume_index",
    "actual_job_role",
    "rank",
    "recommended_job_role"
]

for column in required_semantic_columns:

    if column not in semantic_df.columns:

        raise ValueError(
            f"Missing semantic column: {column}"
        )


# Group each resume's recommendations
semantic_groups = (
    semantic_df
    .sort_values(
        [
            "resume_index",
            "rank"
        ]
    )
    .groupby(
        "resume_index"
    )
)


top1_correct = 0
top3_correct = 0
top5_correct = 0

total_semantic = 0


for resume_index, group in semantic_groups:

    actual_role = str(
        group.iloc[0]["actual_job_role"]
    ).strip().lower()

    predictions = [
        str(role).strip().lower()
        for role in group[
            "recommended_job_role"
        ].tolist()
    ]

    total_semantic += 1

    if len(predictions) >= 1:

        if actual_role == predictions[0]:

            top1_correct += 1

    if actual_role in predictions[:3]:

        top3_correct += 1

    if actual_role in predictions[:5]:

        top5_correct += 1


top1 = (
    top1_correct / total_semantic * 100
)

top3 = (
    top3_correct / total_semantic * 100
)

top5 = (
    top5_correct / total_semantic * 100
)


print(
    f"Top-1 Accuracy: {top1:.2f}%"
)

print(
    f"Top-3 Accuracy: {top3:.2f}%"
)

print(
    f"Top-5 Accuracy: {top5:.2f}%"
)


# ============================================================
# SKILL ALIGNMENT VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("SKILL ALIGNMENT VALIDATION")
print("-" * 70)

skill_column = (
    "skill_alignment_percentage"
)

if skill_column not in skill_df.columns:

    raise ValueError(
        "Skill alignment percentage column not found."
    )


skill_scores = pd.to_numeric(
    skill_df[skill_column],
    errors="coerce"
).dropna()


mean_skill = skill_scores.mean()

median_skill = skill_scores.median()

high_skill = (
    skill_scores >= 75
).mean() * 100

medium_skill = (
    (
        skill_scores >= 50
    )
    &
    (
        skill_scores < 75
    )
).mean() * 100

low_skill = (
    skill_scores < 50
).mean() * 100


print(
    f"Mean Skill Alignment: "
    f"{mean_skill:.2f}%"
)

print(
    f"Median Skill Alignment: "
    f"{median_skill:.2f}%"
)

print(
    f"High Alignment: "
    f"{high_skill:.2f}%"
)

print(
    f"Medium Alignment: "
    f"{medium_skill:.2f}%"
)

print(
    f"Low Alignment: "
    f"{low_skill:.2f}%"
)


# ============================================================
# DATASET STATISTICS
# ============================================================

print("\n" + "-" * 70)
print("DATASET VALIDATION")
print("-" * 70)

if "Job Role" in training_df.columns:

    unique_roles = (
        training_df["Job Role"]
        .nunique()
    )

else:

    unique_roles = 0


if "Skills" in training_df.columns:

    non_empty_skills = (
        training_df["Skills"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .mean()
        * 100
    )

else:

    non_empty_skills = 0


print(
    f"Total records: "
    f"{len(training_df)}"
)

print(
    f"Unique job roles: "
    f"{unique_roles}"
)

print(
    f"Records with skill information: "
    f"{non_empty_skills:.2f}%"
)


# ============================================================
# EXTERNAL BENCHMARK STATUS
# ============================================================

print("\n" + "-" * 70)
print("EXTERNAL BENCHMARK VALIDATION STATUS")
print("-" * 70)

external_benchmark_available = False


print(
    "External benchmark dataset: "
    "NOT AVAILABLE IN PROJECT"
)

print(
    "SemEval career benchmark: "
    "NOT YET CONNECTED"
)

print(
    "Curated LinkedIn transition dataset: "
    "NOT YET CONNECTED"
)

print(
    "\nInternal validation is available "
    "using the project's 10,000-record dataset."
)


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_PATH = os.path.join(
    MODEL_DIR,
    "benchmark_validation_report.txt"
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "CAREER INTELLIGENCE BENCHMARK VALIDATION REPORT\n"
    )

    report.write(
        "=" * 65 + "\n\n"
    )

    report.write(
        "1. INTERNAL DATASET\n"
    )

    report.write(
        f"Records: {len(training_df)}\n"
    )

    report.write(
        f"Unique Job Roles: {unique_roles}\n"
    )

    report.write(
        f"Records With Skills: "
        f"{non_empty_skills:.2f}%\n\n"
    )

    report.write(
        "2. SEMANTIC MATCHING\n"
    )

    report.write(
        f"Top-1 Accuracy: {top1:.2f}%\n"
    )

    report.write(
        f"Top-3 Accuracy: {top3:.2f}%\n"
    )

    report.write(
        f"Top-5 Accuracy: {top5:.2f}%\n\n"
    )

    report.write(
        "3. SKILL ALIGNMENT\n"
    )

    report.write(
        f"Mean Alignment: "
        f"{mean_skill:.2f}%\n"
    )

    report.write(
        f"Median Alignment: "
        f"{median_skill:.2f}%\n"
    )

    report.write(
        f"High Alignment: "
        f"{high_skill:.2f}%\n"
    )

    report.write(
        f"Medium Alignment: "
        f"{medium_skill:.2f}%\n"
    )

    report.write(
        f"Low Alignment: "
        f"{low_skill:.2f}%\n\n"
    )

    report.write(
        "4. EXTERNAL BENCHMARK STATUS\n"
    )

    report.write(
        "SemEval benchmark: Not connected\n"
    )

    report.write(
        "LinkedIn transition dataset: "
        "Not connected\n"
    )

    report.write(
        "\nNote: Internal evaluation results "
        "must not be represented as external "
        "benchmark validation.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK VALIDATION COMPLETED")
print("=" * 70)

print(
    "\nReport saved to:"
)

print(
    REPORT_PATH
)

print("\nDone.")