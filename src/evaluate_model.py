import os
from pathlib import Path
from transformers import Trainer, TrainingArguments

from utils import (
    PROJECT_ROOT,
    load_model_and_tokenizer,
    load_training_dataset,
    tokenize_dataset,
    accuracy_metric,
)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "hf-sentiment-output"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)).resolve()
LOG_DIR = Path(os.environ.get("LOG_DIR", PROJECT_ROOT / "logs")).resolve()


def evaluate_saved_model():
    model_dir = MODEL_DIR if MODEL_DIR.is_dir() else None
    tokenizer, model = load_model_and_tokenizer(str(model_dir) if model_dir else None)
    dataset = load_training_dataset()
    tokenized = tokenize_dataset(dataset, tokenizer)

    print(
        f"📦 Model loaded from: "
        f"{MODEL_DIR if MODEL_DIR.is_dir() else 'Hugging Face cache'}"
    )

    args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "eval-temp"),
        per_device_eval_batch_size=16,
        do_eval=True,
        logging_dir=str(LOG_DIR),
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        args=args,
        compute_metrics=accuracy_metric(),
    )

    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)
    return metrics


if __name__ == "__main__":
    evaluate_saved_model()
