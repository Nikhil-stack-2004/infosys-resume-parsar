import os
import re
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SENTENCE-BERT SEMANTIC SKILL/JOB MATCHING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded: {df.shape[0]} records"
)

print(
    f"Columns: {list(df.columns)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Resume Text",
    "Skills",
    "Job Role"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# PREPARE TEXT
# ============================================================

df["Resume Text"] = (
    df["Resume Text"]
    .fillna("")
    .apply(clean_text)
)

df["Skills"] = (
    df["Skills"]
    .fillna("")
    .apply(clean_text)
)

df["Job Role"] = (
    df["Job Role"]
    .fillna("")
    .astype(str)
)


df["Semantic Text"] = (
    df["Resume Text"]
    + " "
    + df["Skills"]
)


# ============================================================
# LOAD SENTENCE-BERT
# ============================================================

print("\nLoading Sentence-BERT model...")

model_name = "all-MiniLM-L6-v2"

model = SentenceTransformer(
    model_name
)

print(
    f"Model loaded: {model_name}"
)


# ============================================================
# CREATE RESUME/SKILL EMBEDDINGS
# ============================================================

print("\nCreating resume/skill embeddings...")

resume_embeddings = model.encode(
    df["Semantic Text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(
    "\nResume embedding shape:",
    resume_embeddings.shape
)


# ============================================================
# CREATE UNIQUE JOB ROLE TEXT
# ============================================================

job_roles = sorted(
    df["Job Role"].unique()
)

print(
    f"\nUnique job roles: {len(job_roles)}"
)


# ============================================================
# JOB ROLE EMBEDDINGS
# ============================================================

print("\nCreating job-role embeddings...")

job_role_embeddings = model.encode(
    job_roles,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(
    "Job-role embedding shape:",
    job_role_embeddings.shape
)


# ============================================================
# SEMANTIC MATCHING
# ============================================================

print("\nCalculating semantic similarity...")

similarity_matrix = cosine_similarity(
    resume_embeddings,
    job_role_embeddings
)


# ============================================================
# TOP-5 JOB RECOMMENDATIONS
# ============================================================

print("\nGenerating top-5 recommendations...")

recommendations = []

for index in range(
    len(df)
):

    similarities = similarity_matrix[index]

    top_indices = np.argsort(
        similarities
    )[::-1][:5]

    row_recommendations = []

    for rank, role_index in enumerate(
        top_indices,
        start=1
    ):

        role = job_roles[role_index]

        score = (
            similarities[role_index] * 100
        )

        score = max(
            0,
            min(100, score)
        )

        row_recommendations.append(
            {
                "rank": rank,
                "job_role": role,
                "similarity": round(
                    float(score),
                    2
                )
            }
        )

    recommendations.append(
        row_recommendations
    )


# ============================================================
# SAVE TOP-5 RESULTS
# ============================================================

output_rows = []

for index, row_recommendations in enumerate(
    recommendations
):

    for recommendation in row_recommendations:

        output_rows.append(
            {
                "resume_index": index,
                "actual_job_role": df.iloc[index]["Job Role"],
                "rank": recommendation["rank"],
                "recommended_job_role": recommendation["job_role"],
                "semantic_similarity": recommendation["similarity"]
            }
        )


recommendation_df = pd.DataFrame(
    output_rows
)


OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "semantic_job_recommendations.csv"
)

recommendation_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

resume_embedding_path = os.path.join(
    MODEL_DIR,
    "resume_sentence_bert_embeddings.npy"
)

job_embedding_path = os.path.join(
    MODEL_DIR,
    "job_role_sentence_bert_embeddings.npy"
)

np.save(
    resume_embedding_path,
    resume_embeddings
)

np.save(
    job_embedding_path,
    job_role_embeddings
)


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating Top-K recommendations...")

top1_correct = 0
top3_correct = 0
top5_correct = 0

for index, row_recommendations in enumerate(
    recommendations
):

    actual_role = str(
        df.iloc[index]["Job Role"]
    ).strip().lower()

    predicted_roles = [
        str(item["job_role"]).strip().lower()
        for item in row_recommendations
    ]

    if len(predicted_roles) >= 1:
        if actual_role == predicted_roles[0]:
            top1_correct += 1

    if actual_role in predicted_roles[:3]:
        top3_correct += 1

    if actual_role in predicted_roles[:5]:
        top5_correct += 1


total = len(df)

top1_accuracy = (
    top1_correct / total
)

top3_accuracy = (
    top3_correct / total
)

top5_accuracy = (
    top5_correct / total
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    MODEL_DIR,
    "semantic_matching_metrics.txt"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Sentence-BERT Semantic Matching Results\n"
    )

    file.write(
        "=" * 55 + "\n\n"
    )

    file.write(
        f"Model: {model_name}\n"
    )

    file.write(
        f"Dataset records: {total}\n"
    )

    file.write(
        f"Unique job roles: {len(job_roles)}\n\n"
    )

    file.write(
        f"Top-1 Accuracy: "
        f"{top1_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Top-3 Accuracy: "
        f"{top3_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Top-5 Accuracy: "
        f"{top5_accuracy * 100:.2f}%\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SEMANTIC MATCHING COMPLETED")
print("=" * 70)

print(
    f"\nTop-1 Accuracy: "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Top-3 Accuracy: "
    f"{top3_accuracy * 100:.2f}%"
)

print(
    f"Top-5 Accuracy: "
    f"{top5_accuracy * 100:.2f}%"
)

print(
    "\nRecommendations saved:"
)

print(
    OUTPUT_PATH
)

print(
    "\nResume embeddings saved:"
)

print(
    resume_embedding_path
)

print(
    "\nJob-role embeddings saved:"
)

print(
    job_embedding_path
)

print(
    "\nMetrics saved:"
)

print(
    metrics_path
)

print("\nDone.")