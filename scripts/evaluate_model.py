"""Evaluate the career-area model and enforce the CI accuracy gate."""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def evaluate(test_size=0.2, random_state=42):
    data = pd.read_csv(os.path.join(ROOT, "data", "processed", "clean_training_data.csv"))
    data["education_normalized"] = data["Education"].fillna("").str.lower()
    data["combined_text"] = data["Resume Text"].fillna("") + " " + data["Skills"].fillna("")
    _, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data["Category"]
    )

    model = joblib.load(os.path.join(ROOT, "models", "logistic_regression_model.pkl"))
    vectorizer = joblib.load(os.path.join(ROOT, "models", "tfidf_vectorizer.pkl"))
    encoder = joblib.load(os.path.join(ROOT, "models", "education_onehot_encoder.pkl"))
    labels = joblib.load(os.path.join(ROOT, "models", "job_role_label_encoder.pkl"))

    features = hstack([
        vectorizer.transform(test["combined_text"]),
        test[["Experience Years"]].to_numpy(),
        encoder.transform(test[["education_normalized"]]),
    ])
    expected = labels.transform(test["Category"])
    probabilities = model.predict_proba(features)
    top1 = model.classes_[np.argmax(probabilities, axis=1)]
    top3 = model.classes_[np.argsort(probabilities, axis=1)[:, -3:]]
    top3_accuracy = np.mean([actual in choices for actual, choices in zip(expected, top3)])
    return {
        "top1_accuracy": accuracy_score(expected, top1),
        "top3_accuracy": top3_accuracy,
        "classes": len(labels.classes_),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-top1", type=float, default=0.80)
    args = parser.parse_args()
    result = evaluate()
    print(result)
    if result["top1_accuracy"] < args.min_top1:
        raise SystemExit(f"Accuracy gate failed: {result['top1_accuracy']:.3f} < {args.min_top1:.3f}")
