from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")).resolve()
CUSTOM_DATA_PATH = DATA_DIR / "fine_tune_subset.csv"
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "distilbert-base-uncased")
NUM_LABELS = int(os.environ.get("NUM_LABELS", "5"))
MAX_LENGTH = 128


def load_training_dataset(limit: int = 1000, seed: int = 42) -> Dataset:
    """Return a dataset for training, preferring user-provided CSV."""
    if CUSTOM_DATA_PATH.exists():
        df = pd.read_csv(CUSTOM_DATA_PATH)
        return Dataset.from_pandas(df)

    dataset = load_dataset("yelp_review_full", split="train[:2%]")
    dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(min(limit, len(dataset))))


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.train_test_split(test_size=0.2)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


def load_model_and_tokenizer(model_dir: str | None = None) -> Tuple:
    """Load tokenizer and model from disk, or fallback to base model."""
    if model_dir and os.path.isdir(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            DEFAULT_MODEL_NAME,
            num_labels=NUM_LABELS,
        )
    return tokenizer, model


def accuracy_metric():
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    return compute_metrics


def sentiment_from_label(label_id: int):
    if label_id in [0, 1]:
        return 1, "negative"
    if label_id == 2:
        return 2, "neutral"
    return 3, "positive"
