"""
TruthLens Dataset Loader
Downloads and manages raw Kaggle Fake and Real News dataset (Fake.csv & True.csv).
"""

import os
import shutil
import logging
from pathlib import Path
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
FAKE_CSV_PATH = RAW_DATA_DIR / "Fake.csv"
TRUE_CSV_PATH = RAW_DATA_DIR / "True.csv"

# Fallback download URLs if kagglehub is not used
FALLBACK_URLS = {
    "Fake.csv": "https://raw.githubusercontent.com/clmentb/fake-and-real-news-dataset/master/Fake.csv",
    "True.csv": "https://raw.githubusercontent.com/clmentb/fake-and-real-news-dataset/master/True.csv"
}


def download_with_kagglehub() -> bool:
    """Attempt to download Kaggle dataset via kagglehub library."""
    try:
        import kagglehub
        logger.info("Downloading dataset via kagglehub ('clmentbisaillon/fake-and-real-news-dataset')...")
        download_path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        logger.info(f"kagglehub downloaded files to: {download_path}")
        
        # Look for Fake.csv and True.csv in the downloaded directory
        downloaded_dir = Path(download_path)
        found_fake = False
        found_true = False
        
        for file in downloaded_dir.rglob("*.csv"):
            if "fake" in file.name.lower():
                shutil.copy(file, FAKE_CSV_PATH)
                found_fake = True
                logger.info(f"Copied {file.name} to {FAKE_CSV_PATH}")
            elif "true" in file.name.lower():
                shutil.copy(file, TRUE_CSV_PATH)
                found_true = True
                logger.info(f"Copied {file.name} to {TRUE_CSV_PATH}")
                
        return found_fake and found_true
    except Exception as e:
        logger.warning(f"kagglehub download failed or unavailable: {e}")
        return False


def download_file_direct(url: str, dest_path: Path) -> bool:
    """Download a file with stream and progress logging."""
    try:
        logger.info(f"Downloading from {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    
        logger.info(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"Direct download failed for {url}: {e}")
        return False


def download_from_huggingface_fallback() -> bool:
    """Download from Hugging Face dataset repository if primary sources fail."""
    try:
        from datasets import load_dataset
        logger.info("Attempting Hugging Face datasets fallback ('GonzaloA/fake_news')...")
        dataset = load_dataset("GonzaloA/fake_news")
        
        # Combine train, validation, test
        dfs = []
        for split in dataset.keys():
            dfs.append(dataset[split].to_pandas())
        full_df = pd.concat(dfs, ignore_index=True)
        
        fake_df = full_df[full_df['label'] == 1].copy()
        true_df = full_df[full_df['label'] == 0].copy()
        
        fake_df.to_csv(FAKE_CSV_PATH, index=False)
        true_df.to_csv(TRUE_CSV_PATH, index=False)
        logger.info("Successfully fetched dataset from Hugging Face mirror!")
        return True
    except Exception as e:
        logger.warning(f"Hugging Face fallback failed: {e}")
        return False


def ensure_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure Fake.csv and True.csv exist in data/raw.
    Loads and returns (df_fake, df_true).
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not (FAKE_CSV_PATH.exists() and TRUE_CSV_PATH.exists()):
        logger.info("Raw dataset files not found locally. Initiating download...")
        success = download_with_kagglehub()
        
        if not success:
            logger.info("Attempting fallback download method...")
            fake_ok = download_file_direct(FALLBACK_URLS["Fake.csv"], FAKE_CSV_PATH)
            true_ok = download_file_direct(FALLBACK_URLS["True.csv"], TRUE_CSV_PATH)
            
            if not (fake_ok and true_ok):
                logger.info("Attempting Hugging Face dataset fallback...")
                hf_ok = download_from_huggingface_fallback()
                if not hf_ok:
                    raise RuntimeError("Failed to download Fake and Real news dataset from all sources.")
    else:
        logger.info(f"Found existing raw datasets at {FAKE_CSV_PATH} and {TRUE_CSV_PATH}.")
        
    df_fake = pd.read_csv(FAKE_CSV_PATH)
    df_true = pd.read_csv(TRUE_CSV_PATH)
    
    logger.info(f"Loaded Fake news records: {len(df_fake):,}")
    logger.info(f"Loaded Real/True news records: {len(df_true):,}")
    
    return df_fake, df_true


if __name__ == "__main__":
    df_fake, df_true = ensure_dataset()
    print("\n--- Fake Dataset Sample ---")
    print(df_fake.head(2))
    print("\n--- True Dataset Sample ---")
    print(df_true.head(2))
