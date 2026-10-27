"""
TruthLens Explainability Engine (LIME & Feature Attribution)
Explains news authenticity predictions by identifying top words pushing toward Real or Fake.
"""

import sys
import logging
from pathlib import Path
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessor import clean_text_basic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "baseline_pipeline.pkl"


class NewsExplainer:
    """
    LIME-based text explainer for fake news predictions.
    """

    def __init__(self, pipeline_path: Path | str = PIPELINE_PATH):
        pipeline_path = Path(pipeline_path)
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Model pipeline not found at: {pipeline_path}. Run Phase 2 training first.")
            
        logger.info(f"Loading trained pipeline from {pipeline_path}...")
        self.pipeline = joblib.load(pipeline_path)
        self.class_names = ["Real", "Fake"] # 0 = Real, 1 = Fake
        self.explainer = LimeTextExplainer(class_names=self.class_names, split_expression=r"\W+", bow=True)
        logger.info("NewsExplainer initialized successfully.")

    def predict_proba_fn(self, texts: list[str]) -> np.ndarray:
        """
        Internal wrapper to clean texts and predict probabilities via pipeline.
        """
        cleaned = [clean_text_basic(t) for t in texts]
        return self.pipeline.predict_proba(cleaned)

    def explain(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 400
    ) -> dict:
        """
        Generates full prediction and LIME explanation for a single news article/headline.
        """
        if not text or not text.strip():
            return {
                "error": "Input text cannot be empty.",
                "label": "Unknown",
                "confidence_score": 0.0,
                "top_influential_words": []
            }
            
        cleaned_text = clean_text_basic(text)
        if len(cleaned_text.split()) < 3:
            # Fallback for ultra-short texts
            num_samples = min(num_samples, 100)

        # Get base prediction probabilities
        probs = self.predict_proba_fn([text])[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0

        # Run LIME instance explanation
        # Explaining with respect to the Fake class (label 1): positive weight = pushes towards Fake, negative weight = pushes towards Real
        try:
            exp = self.explainer.explain_instance(
                cleaned_text,
                self.predict_proba_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=[1]
            )
            raw_weights = exp.as_list(label=1)
        except Exception as e:
            logger.warning(f"LIME explanation encountered an issue: {e}. Generating fallback attribution...")
            raw_weights = []

        # Format influential words
        influential_words = []
        for word, weight in raw_weights:
            supports = "Fake" if weight > 0 else "Real"
            influential_words.append({
                "word": word,
                "weight": round(float(weight), 4),
                "supports": supports,
                "importance": round(abs(float(weight)), 4)
            })

        # Sort by importance descending
        influential_words.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "label": pred_label,
            "confidence_score": round(confidence, 2),
            "probabilities": {
                "Real": round(float(probs[0]), 4),
                "Fake": round(float(probs[1]), 4)
            },
            "top_influential_words": influential_words,
            "cleaned_text_preview": cleaned_text[:200] + ("..." if len(cleaned_text) > 200 else "")
        }


if __name__ == "__main__":
    explainer = NewsExplainer()

    # Test Sample 1: Authentic / Real News Sample
    real_sample = (
        "WASHINGTON (Reuters) - The U.S. Senate passed a bipartisan spending bill on Thursday "
        "to fund federal agencies and avoid a looming government shutdown, sending the measure "
        "to the White House for the president's signature."
    )
    print("\n--- Explaining REAL News Sample ---")
    res_real = explainer.explain(real_sample, num_features=8)
    print(f"Prediction: {res_real['label']} ({res_real['confidence_score']}%)")
    print("Influential Words:")
    for w in res_real["top_influential_words"]:
        print(f"  • '{w['word']}': weight={w['weight']:+.4f} (supports {w['supports']})")

    # Test Sample 2: Fabricated / Fake News Sample
    fake_sample = (
        "SHOCKING: Secret FBI informant exposes massive underground conspiracy! "
        "Anonymous sources confirm mainstream media coverup of secret election plot."
    )
    print("\n--- Explaining FAKE News Sample ---")
    res_fake = explainer.explain(fake_sample, num_features=8)
    print(f"Prediction: {res_fake['label']} ({res_fake['confidence_score']}%)")
    print("Influential Words:")
    for w in res_fake["top_influential_words"]:
        print(f"  • '{w['word']}': weight={w['weight']:+.4f} (supports {w['supports']})")
