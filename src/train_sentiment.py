import os

from transformers import Trainer, TrainingArguments

from utils import (
    load_training_dataset,
    tokenize_dataset,
    load_model_and_tokenizer,
    accuracy_metric,
)

MODEL_DIR = os.environ.get("OUTPUT_DIR", "hf-sentiment-output")
EPOCHS = int(os.environ.get("EPOCHS", 1))


def main():
    dataset = load_training_dataset()
    tokenizer, model = load_model_and_tokenizer()
    tokenized = tokenize_dataset(dataset, tokenizer)

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=8,
        logging_dir="./logs",
    )

    if hasattr(args, "evaluation_strategy"):
        args.evaluation_strategy = "epoch"
    if hasattr(args, "save_strategy"):
        args.save_strategy = "no"
    if hasattr(args, "report_to"):
        args.report_to = []

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=accuracy_metric(),
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()
