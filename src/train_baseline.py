"""
TruthLens Baseline Model Training & Benchmarking Pipeline
Trains TF-IDF + Logistic Regression, Naive Bayes, and Linear SVM classifiers.
Evaluates test metrics and serializes the best performing model.
"""

import sys
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate_classifier, print_metrics_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test splits from processed directory."""
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    # Fill any potential NaNs in cleaned_text
    train_df["cleaned_text"] = train_df["cleaned_text"].fillna("")
    val_df["cleaned_text"] = val_df["cleaned_text"].fillna("")
    test_df["cleaned_text"] = test_df["cleaned_text"].fillna("")
    
    logger.info(f"Loaded train ({len(train_df):,}), val ({len(val_df):,}), test ({len(test_df):,})")
    return train_df, val_df, test_df


def train_and_benchmark_baselines() -> dict:
    """
    Train TF-IDF vectorizer and 3 baseline classifiers.
    Evaluates on test set and saves best model artifact.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = load_datasets()
    
    X_train = train_df["cleaned_text"].values
    y_train = train_df["label"].values
    
    X_val = val_df["cleaned_text"].values
    y_val = val_df["label"].values
    
    X_test = test_df["cleaned_text"].values
    y_test = test_df["label"].values
    
    # 1. TF-IDF Feature Extraction (Fit ONLY on Train)
    logger.info("Fitting TF-IDF Vectorizer (max_features=50,000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        lowercase=False # Already lowercased in preprocessing
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_):,} features")
    
    # 2. Define Baseline Models
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=3)
    }
    
    all_metrics = []
    trained_models = {}
    
    # 3. Train and Evaluate Each Model
    for name, model in models.items():
        logger.info(f"\n--- Training {name} ---")
        model.fit(X_train_vec, y_train)
        trained_models[name] = model
        
        # Validation Evaluation
        y_val_pred = model.predict(X_val_vec)
        val_acc = np.mean(y_val_pred == y_val)
        logger.info(f"{name} Validation Accuracy: {val_acc:.4f}")
        
        # Test Evaluation
        y_test_pred = model.predict(X_test_vec)
        y_test_prob = model.predict_proba(X_test_vec) if hasattr(model, "predict_proba") else None
        
        metrics = evaluate_classifier(
            y_true=y_test,
            y_pred=y_test_pred,
            y_prob=y_test_prob,
            model_name=name,
            save_cm_plot=True
        )
        all_metrics.append(metrics)

        artifact_name = {
            "Logistic Regression": "logistic_regression_pipeline.pkl",
            "Linear SVM": "svm_pipeline.pkl",
        }.get(name)
        if artifact_name:
            joblib.dump(
                Pipeline([("tfidf", vectorizer), ("classifier", model)]),
                MODELS_DIR / artifact_name,
            )
        
    # 4. Print Summary Benchmark Table
    print_metrics_summary(all_metrics)
    
    # 5. Determine Best Baseline Model (by F1-Score & Accuracy)
    best_metric = max(all_metrics, key=lambda m: (m["f1_score"], m["accuracy"]))
    best_model_name = best_metric["model_name"]
    best_model = trained_models[best_model_name]
    
    logger.info(f"🏆 Best Baseline Model: {best_model_name} (F1: {best_metric['f1_score']:.4f}, Acc: {best_metric['accuracy']:.4f})")
    
    # 6. Save Artifacts
    best_model_path = MODELS_DIR / "best_baseline.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    pipeline_path = MODELS_DIR / "baseline_pipeline.pkl"
    metrics_json_path = MODELS_DIR / "baseline_metrics.json"
    
    joblib.dump(best_model, best_model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # End-to-end inference pipeline
    full_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", best_model)
    ])
    joblib.dump(full_pipeline, pipeline_path)
    
    with open(metrics_json_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "models": all_metrics
        }, f, indent=2)
        
    logger.info(f"Saved best model artifact to: {best_model_path}")
    logger.info(f"Saved TF-IDF vectorizer to: {vectorizer_path}")
    logger.info(f"Saved complete pipeline to: {pipeline_path}")
    logger.info(f"Saved baseline metrics to: {metrics_json_path}")
    
    return {
        "best_model_name": best_model_name,
        "best_metrics": best_metric,
        "all_metrics": all_metrics
    }


if __name__ == "__main__":
    train_and_benchmark_baselines()
