"""
FakeFound Inference & Explainability Service
"""

import sys
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from urllib.parse import urlparse
from pathlib import Path

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability import NewsExplainer
from src.preprocessor import clean_text_basic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGISTIC_PIPELINE_PATH = MODELS_DIR / "logistic_regression_pipeline.pkl"
DISTILBERT_DIR = MODELS_DIR / "distilbert_model"
CREDIBILITY_PATH = PROJECT_ROOT / "data" / "source_credibility.json"


class PredictionService:
    """Singleton service for inference and explanation."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self):
        logger.info("Initializing FakeFound PredictionService...")
        self.explainer = None
        self.logistic_pipeline = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.metrics_data = {}
        self.credibility_domains = {"high": {}, "low": {}}
        self.cache_ttl_seconds = max(1, int(os.getenv("PREDICTION_CACHE_TTL_SECONDS", "300")))
        self.cache_max_entries = max(1, int(os.getenv("PREDICTION_CACHE_MAX_ENTRIES", "256")))
        self.prediction_cache = OrderedDict()
        self.cache_lock = threading.Lock()
        
        # Load Explainer
        try:
            self.explainer = NewsExplainer()
            logger.info("Explainer and baseline pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model pipeline: {e}")

        try:
            if LOGISTIC_PIPELINE_PATH.exists():
                import joblib
                self.logistic_pipeline = joblib.load(LOGISTIC_PIPELINE_PATH)
                logger.info("Logistic Regression pipeline loaded successfully.")
            else:
                logger.warning("Logistic Regression artifact not found at %s.", LOGISTIC_PIPELINE_PATH)
        except Exception as e:
            logger.error(f"Failed to load Logistic Regression pipeline: {e}")

        try:
            if DISTILBERT_DIR.exists():
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.bert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR, local_files_only=True)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                    DISTILBERT_DIR,
                    local_files_only=True,
                )
                self.bert_model.eval()
                logger.info("DistilBERT pipeline loaded successfully.")
            else:
                logger.warning("DistilBERT artifact not found at %s.", DISTILBERT_DIR)
        except Exception as e:
            logger.error(f"Failed to load DistilBERT pipeline: {e}")
            
        # Load Metrics
        if CREDIBILITY_PATH.exists():
            with open(CREDIBILITY_PATH, "r", encoding="utf-8") as f:
                credibility_data = json.load(f)
            self.credibility_domains = {
                "high": {item["domain"].lower(): item["name"] for item in credibility_data.get("high_credibility", [])},
                "low": {item["domain"].lower(): item["name"] for item in credibility_data.get("low_credibility", [])},
            }

        comp_path = MODELS_DIR / "model_comparison.json"
        base_path = MODELS_DIR / "baseline_metrics.json"
        
        if comp_path.exists():
            with open(comp_path, "r") as f:
                self.metrics_data = json.load(f)
        elif base_path.exists():
            with open(base_path, "r") as f:
                self.metrics_data = json.load(f)

    def predict_and_explain(self, text: str, num_explanation_words: int = 8) -> dict:
        """Run prediction and LIME explanation on input text."""
        if not self.explainer:
            raise RuntimeError("Model service is not loaded.")

        cache_key = ("explain", text.strip(), num_explanation_words)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
            
        result = self.explainer.explain(
            text=text,
            num_features=num_explanation_words,
            num_samples=300
        )
        self._set_cached(cache_key, result)
        return result

    def _get_cached(self, key):
        now = time.monotonic()
        with self.cache_lock:
            entry = self.prediction_cache.get(key)
            if not entry:
                return None
            created_at, value = entry
            if now - created_at >= self.cache_ttl_seconds:
                self.prediction_cache.pop(key, None)
                return None
            self.prediction_cache.move_to_end(key)
            return deepcopy(value)

    def _set_cached(self, key, value) -> None:
        with self.cache_lock:
            self.prediction_cache[key] = (time.monotonic(), deepcopy(value))
            self.prediction_cache.move_to_end(key)
            while len(self.prediction_cache) > self.cache_max_entries:
                self.prediction_cache.popitem(last=False)

    @staticmethod
    def _prediction_from_pipeline(pipeline, text: str) -> dict:
        probabilities = pipeline.predict_proba([clean_text_basic(text)])[0]
        label_index = int(np.argmax(probabilities))
        labels = ["Real", "Fake"]
        return {
            "label": labels[label_index],
            "confidence_score": round(float(probabilities[label_index]) * 100, 2),
            "probabilities": {
                "Real": round(float(probabilities[0]), 4),
                "Fake": round(float(probabilities[1]), 4),
            },
        }

    def predict_basic(self, text: str) -> dict:
        """Run the champion model without the slower LIME explanation."""
        if not self.explainer:
            raise RuntimeError("Model service is not loaded.")
        cache_key = ("basic", text.strip())
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        result = self._prediction_from_pipeline(self.explainer.pipeline, text)
        self._set_cached(cache_key, result)
        return result

    def predict_compare(self, text: str) -> list[dict]:
        """Run the available Logistic Regression, SVM, and DistilBERT models."""
        cache_key = ("compare", text.strip())
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        predictions = []
        if self.logistic_pipeline:
            predictions.append({
                "model_name": "Logistic Regression",
                **self._prediction_from_pipeline(self.logistic_pipeline, text),
            })
        if self.explainer:
            predictions.append({
                "model_name": "Linear SVM",
                **self._prediction_from_pipeline(self.explainer.pipeline, text),
            })
        if self.bert_model and self.bert_tokenizer:
            import torch
            encoded = self.bert_tokenizer(
                clean_text_basic(text),
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                probabilities = torch.softmax(self.bert_model(**encoded).logits, dim=1)[0].tolist()
            label_index = int(np.argmax(probabilities))
            predictions.append({
                "model_name": "DistilBERT",
                "label": ["Real", "Fake"][label_index],
                "confidence_score": round(float(probabilities[label_index]) * 100, 2),
                "probabilities": {
                    "Real": round(float(probabilities[0]), 4),
                    "Fake": round(float(probabilities[1]), 4),
                },
            })
        self._set_cached(cache_key, predictions)
        return predictions

    def get_metrics(self) -> dict:
        """Return benchmark metrics across models."""
        return self.metrics_data

    def check_credibility(self, source_url: str) -> dict:
        """Match a URL hostname against the curated source registry."""
        parsed = urlparse(source_url if "://" in source_url else f"https://{source_url}")
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not hostname:
            raise ValueError("A valid URL is required.")
        for rating, domains in self.credibility_domains.items():
            for domain, source_name in domains.items():
                if hostname == domain or hostname.endswith(f".{domain}"):
                    return {
                        "domain": hostname,
                        "rating": "High" if rating == "high" else "Low",
                        "source_name": source_name,
                        "message": "Listed in the curated source registry; this is a signal, not a fact check.",
                    }
        return {
            "domain": hostname,
            "rating": "Unknown",
            "source_name": None,
            "message": "This domain is not in the curated source registry.",
        }

    def is_ready(self) -> bool:
        """Check if pipeline is loaded and ready for inference."""
        return self.explainer is not None
