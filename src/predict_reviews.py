from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from transformers import pipeline

from utils import DEFAULT_MODEL_NAME, sentiment_from_label

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DEFAULT_MODEL_DIR = PROJECT_ROOT / "hf-sentiment-output"

MODEL_DIR = os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))
INPUT_PATH = os.environ.get("INPUT_PATH", str(PROJECT_ROOT / "data" / "test_reviews.csv"))
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", str(PROJECT_ROOT / "data" / "review_predictions.csv"))


if os.path.isdir(MODEL_DIR):
    model_path = MODEL_DIR
else:
    model_path = DEFAULT_MODEL_NAME

classifier = pipeline("text-classification", model=model_path)


def normalize_label(label_str: str) -> int:
    if label_str.startswith("LABEL_"):
        return int(label_str.split("_")[-1])
    return int(label_str)


def predict_reviews(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    predictions = []

    for _, row in df.iterrows():
        text = row["review"]
        pred = classifier(text)[0]
        label_id = normalize_label(pred["label"])
        confidence = round(float(pred["score"]), 3)
        rating, sentiment = sentiment_from_label(label_id)
        predictions.append(
            {
                "id": row.get("id"),
                "review": text,
                "label_id": label_id,
                "sentiment": sentiment,
                "rating": rating,
                "confidence": confidence,
            }
        )

    result = pd.DataFrame(predictions)
    result.to_csv(output_csv, index=False)
    print(result)


if __name__ == "__main__":
    predict_reviews(INPUT_PATH, OUTPUT_PATH)
