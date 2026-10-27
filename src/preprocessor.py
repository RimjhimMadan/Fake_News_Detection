"""
TruthLens Text Preprocessing & Data Splitting Pipeline
Cleans raw text, handles duplicates/missing values, and generates stratified train/val/test splits.
"""

import os
import sys
import re
import string
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path("data/processed")


def download_nltk_resources():
    """Ensure required NLTK tokenizers, stopwords, and lemmatizers are downloaded."""
    resources = ["stopwords", "wordnet", "omw-1.4", "punkt"]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK resource {res}: {e}")


def clean_text_basic(text: str) -> str:
    """
    Standard text cleaner:
    - Lowercases text
    - Removes URLs, HTML tags, emails
    - Removes punctuation and extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    
    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)
    
    # Remove brackets / square brackets content often found in Reuters/news tags like "[Reuters]" or "(Reuters)"
    text = re.sub(r"\[.*?\]", " ", text)
    
    # Remove non-alphanumeric characters (keep basic word chars & spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


class TextPreprocessor:
    """Full NLP Preprocessing pipeline with stopword removal and lemmatization."""

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        download_nltk_resources()
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()
            
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def preprocess_single(self, text: str) -> str:
        """Clean a single text string."""
        cleaned = clean_text_basic(text)
        tokens = cleaned.split()
        
        if self.remove_stopwords and self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
            
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            
        return " ".join(tokens)

    def process_dataframe(self, df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
        """Preprocess a dataframe's text column."""
        logger.info(f"Preprocessing {len(df):,} records...")
        df = df.copy()
        df["cleaned_text"] = df[text_col].apply(self.preprocess_single)
        # Filter out rows that became empty after cleaning
        df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)
        return df


def prepare_and_split_data(
    df_fake: pd.DataFrame,
    df_true: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merges Fake (label=1) and True (label=0) dataframes, cleans, deduplicates,
    and performs a stratified train/val/test split.
    """
    logger.info("Merging and formatting raw datasets...")
    
    # Assign labels: 1 = Fake, 0 = Real / True
    df_fake = df_fake.copy()
    df_fake["label"] = 1
    df_fake["label_name"] = "Fake"
    
    df_true = df_true.copy()
    df_true["label"] = 0
    df_true["label_name"] = "Real"
    
    # Concatenate
    df = pd.concat([df_fake, df_true], ignore_index=True)
    logger.info(f"Total raw combined records: {len(df):,}")
    
    # Combine title and text for rich context
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["full_text"] = df["title"] + " " + df["text"]
    
    # Drop duplicates on full_text to prevent leakage
    initial_len = len(df)
    df = df.drop_duplicates(subset=["full_text"]).reset_index(drop=True)
    logger.info(f"Dropped {initial_len - len(df):,} duplicate records. Remaining: {len(df):,}")
    
    # Preprocess text
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    df = preprocessor.process_dataframe(df, text_col="full_text")
    logger.info(f"Records after cleaning & non-empty filtering: {len(df):,}")
    
    # Stratified Train / Val / Test Split (70 / 15 / 15)
    # First split: train (70%) vs temp (30%)
    df_train, df_temp = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["label"],
        random_state=random_state
    )
    
    # Second split: val (15%) and test (15%) from temp
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        stratify=df_temp["label"],
        random_state=random_state
    )
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    logger.info(f"Saved Train set ({len(df_train):,} rows, {len(df_train)/len(df)*100:.1f}%) to {train_path}")
    logger.info(f"Saved Validation set ({len(df_val):,} rows, {len(df_val)/len(df)*100:.1f}%) to {val_path}")
    logger.info(f"Saved Test set ({len(df_test):,} rows, {len(df_test)/len(df)*100:.1f}%) to {test_path}")
    
    return df_train, df_val, df_test


if __name__ == "__main__":
    from src.data_loader import ensure_dataset
    df_fake, df_true = ensure_dataset()
    train_df, val_df, test_df = prepare_and_split_data(df_fake, df_true)
