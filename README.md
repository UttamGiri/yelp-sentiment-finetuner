# 🧠 Hugging Face Sentiment Fine-Tuning

## Overview

This project fine-tunes a free Hugging Face model (`distilbert-base-uncased`) on a small subset of Yelp reviews to classify sentiment. Everything runs locally or inside Docker on CPU hardware so the total cost stays at $0.

### 🔹 Tasks

1. Fine-tune pre-trained DistilBERT on Yelp review data.
2. Evaluate accuracy.
3. Predict sentiment for sample reviews and assign a 1–3 rating.
4. Package everything in Docker for reproducibility.

---

## 🏗️ Project Structure

```
huggingface-sentiment-demo/
├── data/
│   ├── test_reviews.csv
│   └── fine_tune_subset.csv
├── src/
│   ├── train_sentiment.py
│   ├── evaluate_model.py
│   ├── predict_reviews.py
│   └── utils.py
├── requirements.txt
├── Dockerfile
└── README.md
```

- `train_sentiment.py` – launches fine-tuning and saves the model to `hf-sentiment-output/`.
- `evaluate_model.py` – benchmarks a saved model using accuracy.
- `predict_reviews.py` – runs inference against any CSV (defaults to `data/test_reviews.csv`).
- `utils.py` – shared helpers for dataset loading, tokenization, metrics, and label mapping.

---

## ⚙️ Setup (Local)

```bash
pip install -r requirements.txt
python src/train_sentiment.py
python src/evaluate_model.py
python src/predict_reviews.py
```

### 🐳 Setup (Docker)

Build and run:

```bash
docker build -t sentiment-demo .
docker run -it --rm sentiment-demo
```

After training completes, you’ll have:

```
/app/hf-sentiment-output/
```

containing the fine-tuned model and tokenizer ready for inference.

---

## 📈 Dataset

Uses Hugging Face `yelp_review_full` (subset of 1,000 samples by default). Classes:

- `0` = very negative
- `1` = negative
- `2` = neutral
- `3` = positive
- `4` = very positive

Include your own CSV at `data/fine_tune_subset.csv` to override the default dataset.

---

## 🔁 Compare Before vs After

Follow this quick loop to measure improvement:

1. **Prepare evaluation set** – `data/test_reviews.csv` contains five Yelp-style reviews that act as a sanity check.
2. **Baseline predictions** – run the pipeline without a saved model:
   ```bash
   export INPUT_PATH=../data/test_reviews.csv
   export OUTPUT_PATH=../data/test_predictions_before.csv
   python src/predict_reviews.py
   ```
   With no `hf-sentiment-output/` directory yet, the script falls back to the base `distilbert-base-uncased` weights. Expect mostly random labels.
3. **Fine-tune** – train on your labeled subset:
   ```bash
   python src/train_sentiment.py
   ```
   This writes the personalized checkpoint to `hf-sentiment-output/`.
4. **After predictions** – run inference again and write a second CSV:
   ```bash
   export INPUT_PATH=../data/test_reviews.csv
   export OUTPUT_PATH=../data/test_predictions_after.csv
   python src/predict_reviews.py
   ```
   The script automatically detects the fine-tuned model directory and loads it.
5. **Compare outputs** – inspect the two CSVs side by side to see how accuracy jumps from random to sentiment-aware.

---

## 🧩 Before Fine-Tuning

- Model outputs nearly random class IDs (0–4) for any text.
- Probabilities are ~uniform (~20% each).
- Accuracy on held-out test set ≈ 0.20 (random guessing).
- Example: `"Excellent service and friendly staff"` → class_id: `1` (random).

---

## ✅ After Fine-Tuning

- Model learns polarity of words like *excellent*, *terrible*, *not bad*.
- Probabilities are sharp and meaningful.
- Accuracy ≈ 0.85–0.90 on the small dataset.
- Example: `"Excellent service and friendly staff"` → class_id: `4` (very positive, 0.93 confidence).

| Example Review                              | Before FT | After FT  |
|---------------------------------------------|-----------|-----------|
| "Not bad at all, works better than expected." | Random    | Positive  |
| "Terrible app experience."                 | Random    | Very Negative |
| "Average service."                         | Random    | Neutral   |

---

## 💾 Outputs

| File                               | Description                                                     |
|------------------------------------|-----------------------------------------------------------------|
| `hf-sentiment-output/`             | Fine-tuned model & tokenizer                                    |
| `data/test_reviews.csv`            | Shared evaluation set for before/after comparisons              |
| `data/test_predictions_before.csv` | Predictions from the base `distilbert-base-uncased` checkpoint |
| `data/test_predictions_after.csv`  | Predictions from the fine-tuned model                           |
| `data/review_predictions.csv`      | Default output path when no custom filenames are provided       |

Example (after fine-tuning):

```
id,review,label_id,sentiment,rating,confidence
1,Excellent food and service!,4,positive,3,0.99
2,Not bad, could be better.,2,neutral,2,0.87
3,Terrible app experience, very buggy.,0,negative,1,0.99
```

---

## 📊 Expected Metrics

| Stage               | Accuracy | Behavior             |
|---------------------|----------|----------------------|
| Before fine-tuning  | ~0.20    | Random output        |
| After fine-tuning   | ~0.88    | Learns correct sentiment |

Run `python src/evaluate_model.py` after training to verify your own metrics.

---

## 🧠 Next Steps

- Add TensorBoard for training visualization.
- Integrate with OpenTelemetry or Jaeger to trace fine-tuning events.
- Replace the dataset with your own labeled text (CSV).
- Deploy the model as a FastAPI inference endpoint.

---

## 💰 Cost

All components are open source and run on local CPU → total cost **$0**.

---

## 🧩 Execution Instructions

| Task                           | Command |
|--------------------------------|---------|
| Train (local)                  | `python src/train_sentiment.py` |
| Evaluate model                 | `python src/evaluate_model.py` |
| Predict (default path)         | `python src/predict_reviews.py` |
| Predict custom CSV             | `INPUT_PATH=../data/test_reviews.csv OUTPUT_PATH=../data/test_predictions_before.csv python src/predict_reviews.py` |
| Train in Docker                | `docker run -it --rm sentiment-demo` |
| Fine-tuned model output        | `hf-sentiment-output/` |

---

## 🧠 What to Expect

| Scenario              | Output                                           | Notes                                                 |
|-----------------------|--------------------------------------------------|-------------------------------------------------------|
| Without Fine-Tuning   | Model guesses random labels (≈20% accuracy)      | Pretrained model doesn’t understand sentiment context.|
| After Fine-Tuning     | Correctly classifies positive/neutral/negative (≈90% accuracy) | Learns sentiment-specific patterns, context, and negation. |


🧠 1️⃣ What distilbert-base-uncased actually is

It’s a pre-trained Transformer model released by Hugging Face.

Architecture: DistilBERT, a smaller (“distilled”) version of BERT.

“Uncased” = all text is lower-cased during training and inference (so “Good” = “good”).

Trained originally on English Wikipedia + Toronto BookCorpus for the general task of masked-language modeling (predicting missing words).

So out of the box, it’s a language understanding model, not yet a sentiment classifier.

⚙️ 2️⃣ What you do during fine-tuning

Your script:

Downloads that general-purpose model’s weights.

Adds a new classification head (a small linear layer on top).

Feeds labeled Yelp reviews (text + sentiment label 0-4) through it.

Updates the weights slightly so it becomes good at mapping a review → sentiment label.

That process is fine-tuning a general LLM on a specific downstream task.

🔄 3️⃣ How it behaves before vs after fine-tuning
Stage	Model knowledge	Behavior
Before fine-tune	Knows English grammar & word relationships	Can’t tell if text is positive/negative → predictions ~random
After fine-tune	Remembers English + learns which patterns signal emotion	Correctly classifies “excellent”, “terrible”, “not bad” etc.

So you’ve turned a general LLM into a task-specific classifier.

🧩 4️⃣ Is DistilBERT a “large language model (LLM)”?

Yes — technically it’s an LLM architecture, but a small one:

≈ 66 million parameters (vs GPT-3’s 175 billion).

Optimized for NLP understanding tasks (sentiment, classification, NER, QA).

Does not generate long text; it’s a BERT-style encoder, not a generative decoder like GPT.

So:

🔹 GPT = text-generation LLM (decoder)
🔹 BERT/DistilBERT = text-understanding LLM (encoder)

🧩 5️⃣ Why DistilBERT is used here
Reason	Benefit
Lightweight	Runs fast on CPU – great for demos
Open-source	Free, no API key needed
High accuracy	85-90 % with small fine-tune set
Easy to deploy	Fits easily in Docker / EKS
Compatible	Works with Hugging Face Trainer out-of-the-box
🧠 6️⃣ Summary
Concept	Meaning
Base model	distilbert-base-uncased (general English understanding)
Fine-tuning task	Sentiment classification on Yelp reviews
LLM type	Encoder-only Transformer
Before fine-tune	Random sentiment guesses
After fine-tune	Learns polarity patterns, outputs accurate ratings


The first time your script runs, Hugging Face will:

Check your local cache (~/.cache/huggingface/transformers/).

If the model isn’t there, it automatically downloads it once from the Hugging Face Hub.

Saves it locally for all future runs.

So you only need internet access the first time.
After that, it loads instantly from your local disk.

⚙️ Where it’s stored

Default cache path (macOS / Linux):

~/.cache/huggingface/hub/models--distilbert-base-uncased/


Inside, you’ll find:

config.json
pytorch_model.bin
tokenizer.json
vocab.txt
tokenizer_config.json


Together they’re about 250 MB total.

💾 How it gets there (automatically)

Example line from your training script:

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)


The .from_pretrained() call triggers the one-time download if needed.


NOTE:

Transformer → The idea or architecture that makes modern AI powerful.

PyTorch → The most popular framework for building transformers.

TensorFlow → Another framework, more common in production and mobile apps.

python -m venv .venv
Create isolated Python environment


Hugging Face loaded the base model (distilbert-base-uncased) — not your fine-tuned one.

The classification head (the last layer that maps embeddings → sentiment labels) is newly initialized with random weights, because the base model doesn’t include your sentiment task.

It’s just warning you:

“You haven’t trained this layer yet — so your predictions will be random until you fine-tune.”

✅ This is normal if you haven’t run train_sentiment.py yet.


OUTPUT WHEN UNTRAINED

python3 src/predict_reviews.py

You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Device set to use mps:0
   id                                     review  label_id sentiment  rating  confidence
0   1                Excellent food and service!         1  negative       1       0.520
1   2                  Not bad, could be better.         1  negative       1       0.523
2   3       Terrible app experience, very buggy.         1  negative       1       0.536
3   4          Average quality, nothing special.         1  negative       1       0.516
4   5  Great customer support and fast shipping!         1  negative       1       0.531

TRAIN

python3 src/train_sentiment.py 
python3 src/predict_reviews.py

Now result is different 

SINGLE TIME TRAINED

   id                                     review  label_id sentiment  rating  confidence
0   1                Excellent food and service!         4  positive       3       0.223
1   2                  Not bad, could be better.         4  positive       3       0.217
2   3       Terrible app experience, very buggy.         4  positive       3       0.225
3   4          Average quality, nothing special.         4  positive       3       0.226
4   5  Great customer support and fast shipping!         4  positive       3       0.227

TRAIN AGAIN  TIME

python3 src/train_sentiment.py 
python3 src/predict_reviews.py

   id                                     review  label_id sentiment  rating  confidence
0   1                Excellent food and service!         1  negative       1       0.218
1   2                  Not bad, could be better.         1  negative       1       0.227
2   3       Terrible app experience, very buggy.         1  negative       1       0.239
3   4          Average quality, nothing special.         2   neutral       2       0.227
4   5  Great customer support and fast shipping!         1  negative       1       0.229

EPOCHS=30 python3 src/train_sentiment.py #means trains for 30 times
python3 src/evaluate_model.py


Metric	Before training	After 3 epochs (you saw)	Expected after 30 epochs
eval_loss	~1.58	0.88	≈ 0.4 – 0.6
eval_accuracy	0.0	1.0 (tiny dataset)	≈ 0.9 – 0.95 on larger data

AFTER TRAI NING

'eval_loss': 0.8059847354888916, 'eval_model_preparation_time': 0.0012, 'eval_accuracy': 0.73, 'eval_runtime': 18.1365, 'eval_samples_per_second': 11.027, 'eval_steps_per_second': 0.717

0   1                Excellent food and service!         4  positive       3       0.470
1   2                  Not bad, could be better.         2   neutral       2       0.427
2   3       Terrible app experience, very buggy.         0  negative       1       0.553
3   4          Average quality, nothing special.         1  negative       1       0.523
4   5  Great customer support and fast shipping!         4  positive       3       0.497


1️⃣ eval_loss — the error value (lower is better)
Model state	Typical eval_loss range	Meaning
🚫 Untrained / Random	1.8 – 2.5	Model is just guessing — predictions are nearly random across 5 labels.
⚙️ Partially trained (few epochs)	1.0 – 1.5	Model starting to recognize positive/negative words.
✅ Well trained	0.6 – 0.9	Model correctly classifies most examples with good confidence.
🌟 Highly fine-tuned	0.3 – 0.6	Model is confident, well-optimized — ideal range for production.
⚠️ Overfitting (too low + high train accuracy)	< 0.2	Model memorized training data; may not generalize well.

📉 Rule of thumb:

For classification (5 labels, like Yelp Review Full), a loss between 0.5–0.8 is strong performance.

🎯 2️⃣ eval_accuracy — the correctness value (higher is better)
Model state	Typical eval_accuracy range	Meaning
🚫 Untrained / Random	0.0 – 0.25	Model is guessing randomly (20% chance for 5 labels).
⚙️ Lightly trained (few epochs)	0.5 – 0.7	Model picks up sentiment cues but still misses subtleties.
✅ Well trained	0.7 – 0.85	Good generalization, consistent accuracy.
🌟 Highly fine-tuned / near production	0.85 – 0.95+	Excellent accuracy; model distinguishes nuanced sentiment.
⚠️ Too high + rising loss	> 0.98	Might indicate overfitting on small dataset.

📈 Rule of thumb:

For 5-class sentiment analysis, 70–85% accuracy after fine-tuning is realistic; 90%+ is exceptional.


 TERMINAL ENTER :   du -sh ~/.cache/huggingface/datasets/yelp_review_full/
 OUTPUT 

499M    /Users/uttamgiri/.cache/huggingface/datasets/yelp_review_full/# yelp-sentiment-finetuner
