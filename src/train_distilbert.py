"""
TruthLens Advanced Transformer Model (DistilBERT Fine-Tuning) Pipeline
Fine-tunes distilbert-base-uncased for fake news detection, evaluates on test data,
and compares metrics against classical baseline models.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
DISTILBERT_DIR = MODELS_DIR / "distilbert_model"
PROCESSED_DATA_DIR = Path("data/processed")


class NewsTextDataset(Dataset):
    """PyTorch Dataset for tokenized news articles."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def train_distilbert(
    model_name: str = "distilbert-base-uncased",
    max_length: int = 64,
    batch_size: int = 32,
    epochs: int = 2,
    lr: float = 3e-5,
    sample_train_size: int = 800,
    sample_test_size: int = 400,
    random_state: int = 42
) -> dict:
    """
    Fine-tune DistilBERT for binary classification and evaluate against baselines.
    """
    DISTILBERT_DIR.mkdir(parents=True, exist_ok=True)
    import os
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} (threads: {torch.get_num_threads()})")

    # 1. Load Data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv").dropna(subset=["cleaned_text"])
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv").dropna(subset=["cleaned_text"])
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv").dropna(subset=["cleaned_text"])

    # For efficient CPU fine-tuning, sample a stratified subset using train_test_split
    from sklearn.model_selection import train_test_split
    if len(train_df) > sample_train_size and device.type == "cpu":
        logger.info(f"Sampling stratified {sample_train_size:,} records for efficient CPU fine-tuning...")
        train_sample, _ = train_test_split(
            train_df,
            train_size=sample_train_size,
            stratify=train_df["label"],
            random_state=random_state
        )
    else:
        train_sample = train_df

    if len(test_df) > sample_test_size and device.type == "cpu":
        test_sample, _ = train_test_split(
            test_df,
            train_size=sample_test_size,
            stratify=test_df["label"],
            random_state=random_state
        )
    else:
        test_sample = test_df

    logger.info(f"Training on {len(train_sample):,} samples, testing on {len(test_sample):,} samples.")

    # 2. Tokenizer & Datasets
    logger.info(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = NewsTextDataset(train_sample["cleaned_text"].tolist(), train_sample["label"].tolist(), tokenizer, max_length)
    test_dataset = NewsTextDataset(test_sample["cleaned_text"].tolist(), test_sample["label"].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model Initialization
    logger.info(f"Loading model architecture: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 4. Training Loop
    logger.info("Starting DistilBERT fine-tuning loop...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_epoch_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} completed | Average Loss: {avg_epoch_loss:.4f}")

    # 5. Evaluation on Test Set
    logger.info("Evaluating DistilBERT on test set...")
    model.eval()
    all_preds = []
    all_probs = []
    all_trues = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_trues.extend(labels.cpu().numpy())

    distilbert_metrics = evaluate_classifier(
        y_true=np.array(all_trues),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        model_name="DistilBERT (Fine-Tuned)",
        save_cm_plot=True
    )

    # 6. Save Model & Tokenizer
    logger.info(f"Saving fine-tuned DistilBERT weights & tokenizer to {DISTILBERT_DIR}...")
    model.save_pretrained(DISTILBERT_DIR)
    tokenizer.save_pretrained(DISTILBERT_DIR)

    # 7. Model Comparison Against Baselines
    baseline_metrics_path = MODELS_DIR / "baseline_metrics.json"
    baseline_models = []
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r") as f:
            baseline_data = json.load(f)
            baseline_models = baseline_data.get("models", [])

    all_compared_models = baseline_models + [distilbert_metrics]

    # Save complete model comparison JSON
    comparison_json_path = MODELS_DIR / "model_comparison.json"
    with open(comparison_json_path, "w") as f:
        json.dump({
            "models": all_compared_models,
            "distilbert_metrics": distilbert_metrics
        }, f, indent=2)

    # Print Final Comparative Benchmark Table
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rows = []
    for m in all_compared_models:
        auc_str = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "N/A"
        rows.append([
            m["model_name"],
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            auc_str
        ])

    df_comp = pd.DataFrame(rows, columns=headers)
    print("\n" + "="*80)
    print(" COMPREHENSIVE MODEL BENCHMARK: BASELINES VS TRANSFORMER (DISTILBERT)")
    print("="*80)
    print(df_comp.to_string(index=False))
    print("="*80 + "\n")

    return {
        "distilbert_metrics": distilbert_metrics,
        "all_models": all_compared_models
    }


if __name__ == "__main__":
    train_distilbert()
