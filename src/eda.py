"""
TruthLens Exploratory Data Analysis (EDA) Engine
Generates descriptive statistics, class balance metrics, text length distributions,
most common n-grams per class, and exports visualizations.
"""

import sys
from pathlib import Path
from collections import Counter
import logging

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PLOTS_DIR = Path("notebooks/plots")


def run_eda(train_path: str = "data/processed/train.csv") -> dict:
    """Execute full EDA on processed training data and export visualizations."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    logger.info(f"Loading dataset from {train_path} for EDA...")
    df = pd.read_csv(train_path)
    
    # 1. Class Distribution
    class_counts = df["label_name"].value_counts()
    class_percentages = df["label_name"].value_counts(normalize=True) * 100
    
    print("\n" + "="*50)
    print(" 1. CLASS BALANCE ANALYSIS")
    print("="*50)
    for label, count in class_counts.items():
        print(f" - {label:<10}: {count:,} samples ({class_percentages[label]:.2f}%)")
        
    # Plot Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=["#e74c3c", "#2ecc71"], ax=ax)
    ax.set_title("Class Balance (Real vs Fake News)", fontsize=14, pad=12, fontweight="bold")
    ax.set_ylabel("Number of Articles", fontsize=11)
    ax.set_xlabel("Authenticity Label", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    class_plot_path = PLOTS_DIR / "eda_class_distribution.png"
    plt.savefig(class_plot_path, dpi=150)
    plt.close()
    
    # 2. Text Length and Word Count Analysis
    df["char_length"] = df["cleaned_text"].astype(str).apply(len)
    df["word_count"] = df["cleaned_text"].astype(str).apply(lambda x: len(x.split()))
    
    print("\n" + "="*50)
    print(" 2. TEXT LENGTH & WORD COUNT STATISTICS")
    print("="*50)
    stats_df = df.groupby("label_name")[["word_count", "char_length"]].agg(["mean", "median", "std", "min", "max"])
    print(stats_df.round(2))
    
    # Plot Word Count Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x="word_count", hue="label_name", kde=True, bins=50,
                 palette={"Fake": "#e74c3c", "Real": "#2ecc71"}, ax=axes[0], alpha=0.5)
    axes[0].set_xlim(0, 1500)
    axes[0].set_title("Word Count Distribution by Class (Capped at 1,500 words)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Word Count per Article", fontsize=11)
    
    sns.boxplot(data=df, x="label_name", y="word_count", palette=["#e74c3c", "#2ecc71"], ax=axes[1], showfliers=False)
    axes[1].set_title("Word Count Boxplot (Outliers Excluded)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Authenticity Label", fontsize=11)
    axes[1].set_ylabel("Word Count", fontsize=11)
    
    plt.tight_layout()
    length_plot_path = PLOTS_DIR / "eda_text_length_distribution.png"
    plt.savefig(length_plot_path, dpi=150)
    plt.close()
    
    # 3. Top Most Frequent Words per Class
    print("\n" + "="*50)
    print(" 3. TOP 15 MOST FREQUENT WORDS PER CLASS")
    print("="*50)
    
    fake_words = " ".join(df[df["label"] == 1]["cleaned_text"].dropna()).split()
    real_words = " ".join(df[df["label"] == 0]["cleaned_text"].dropna()).split()
    
    fake_counter = Counter(fake_words).most_common(15)
    real_counter = Counter(real_words).most_common(15)
    
    print("Top Words in FAKE news:")
    for word, freq in fake_counter[:10]:
        print(f"  - {word:<15}: {freq:,}")
        
    print("\nTop Words in REAL news:")
    for word, freq in real_counter[:10]:
        print(f"  - {word:<15}: {freq:,}")
        
    # Plot Top Words
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    fake_df_words = pd.DataFrame(fake_counter, columns=["word", "count"])
    sns.barplot(data=fake_df_words, y="word", x="count", color="#e74c3c", ax=axes[0])
    axes[0].set_title("Top 15 Words in FAKE News", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Frequency", fontsize=11)
    
    real_df_words = pd.DataFrame(real_counter, columns=["word", "count"])
    sns.barplot(data=real_df_words, y="word", x="count", color="#2ecc71", ax=axes[1])
    axes[1].set_title("Top 15 Words in REAL News", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Frequency", fontsize=11)
    
    plt.tight_layout()
    top_words_path = PLOTS_DIR / "eda_top_words.png"
    plt.savefig(top_words_path, dpi=150)
    plt.close()
    
    print("\n" + "="*50)
    print(f" EDA Complete! Visualizations saved in: {PLOTS_DIR}/")
    print("="*50)
    
    return {
        "class_counts": class_counts.to_dict(),
        "stats": stats_df.to_dict(),
        "top_fake_words": fake_counter,
        "top_real_words": real_counter,
        "plots": [str(class_plot_path), str(length_plot_path), str(top_words_path)]
    }


if __name__ == "__main__":
    run_eda()
