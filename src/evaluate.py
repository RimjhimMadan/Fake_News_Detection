"""
TruthLens Evaluation Utilities
Computes metrics (Accuracy, Precision, Recall, F1, ROC-AUC), classification reports,
and generates confusion matrix visualizations.
"""

import sys
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PLOTS_DIR = Path("notebooks/plots")


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "Classifier",
    save_cm_plot: bool = True
) -> dict:
    """
    Computes standard classification metrics and optionally saves confusion matrix plot.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary")
    rec = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    
    auc = None
    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                prob_pos = y_prob[:, 1]
            else:
                prob_pos = y_prob
            auc = roc_auc_score(y_true, prob_pos)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC for {model_name}: {e}")
            
    cm = confusion_matrix(y_true, y_pred)
    
    # Save Confusion Matrix Plot
    cm_plot_path = None
    if save_cm_plot:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            xticklabels=["Real (0)", "Fake (1)"],
            yticklabels=["Real (0)", "Fake (1)"],
            cbar=False,
            ax=ax
        )
        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        plt.tight_layout()
        
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        cm_plot_path = PLOTS_DIR / f"confusion_matrix_{safe_name}.png"
        plt.savefig(cm_plot_path, dpi=150)
        plt.close()
        
    metrics = {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "cm_plot": str(cm_plot_path) if cm_plot_path else None
    }
    
    return metrics


def print_metrics_summary(metrics_list: list[dict]):
    """Print formatted comparative metrics table."""
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rows = []
    for m in metrics_list:
        auc_str = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "N/A"
        rows.append([
            m["model_name"],
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            auc_str
        ])
        
    df_table = pd.DataFrame(rows, columns=headers)
    print("\n" + "="*75)
    print(" BASELINE MODELS EVALUATION BENCHMARK (TEST SET)")
    print("="*75)
    print(df_table.to_string(index=False))
    print("="*75 + "\n")
    return df_table
