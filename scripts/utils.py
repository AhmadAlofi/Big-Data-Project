"""
utils.py - Helper functions for Arabic AI-text detection project
Contains common utilities, configuration, and shared functions
"""

import os
import re
import subprocess
import sys
import nltk
from typing import List, Set, Tuple
import pandas as pd


# ============================================================================
# Environment Setup
# ============================================================================

def create_project_structure(base_dir: str = "~/arabic_ai_detection") -> dict:
    """
    Create project directory structure if it doesn't exist.
    
    Args:
        base_dir: Root directory for the project (expanduser applied)
        
    Returns:
        Dictionary with all directory paths
    """
    base_dir = os.path.expanduser(base_dir)
    
    dirs = {
        "PROJECT_ROOT": base_dir,
        "DATA_RAW": os.path.join(base_dir, "data", "raw"),
        "DATA_PROCESSED": os.path.join(base_dir, "data", "processed"),
        "MODELS_DIR": os.path.join(base_dir, "models"),
        "REPORTS_DIR": os.path.join(base_dir, "reports", "figures"),
        "NOTEBOOKS_DIR": os.path.join(base_dir, "notebooks"),
        "SCRIPTS_DIR": os.path.join(base_dir, "scripts"),
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    return dirs


def install_packages(packages: List[str] = None):
    """
    Install required Python libraries if not already present.
    
    Args:
        packages: List of package names to install (uses default if None)
    """
    if packages is None:
        packages = [
            "datasets", "huggingface_hub", "pyarabic", "regex",
            "nltk", "wordcloud", "seaborn", "pyarrow"
        ]
    
    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg,
             "--trusted-host", "pypi.org",
             "--trusted-host", "files.pythonhosted.org"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    # Download NLTK resources
    nltk.download("stopwords", quiet=True)


# ============================================================================
# Text Processing Functions
# ============================================================================

def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritical marks.
    
    Args:
        text: Input Arabic text
        
    Returns:
        Text with diacritics removed
    """
    if not isinstance(text, str):
        return ""
    arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u065F]")
    return arabic_diacritics.sub("", text)


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic characters (standardize Alef, Yeh, Teh).
    
    Args:
        text: Input Arabic text
        
    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """
    Remove Arabic stopwords using NLTK's Arabic stopword list.
    
    Args:
        text: Input text
        
    Returns:
        Text with stopwords removed
    """
    from nltk.corpus import stopwords
    if not isinstance(text, str):
        return ""
    ar_stopwords = set(stopwords.words("arabic"))
    return " ".join(w for w in text.split() if w not in ar_stopwords)


def stem_arabic(text: str) -> str:
    """
    Apply Arabic stemming using ISRIStemmer.
    
    Args:
        text: Input text
        
    Returns:
        Stemmed text
    """
    try:
        from nltk.stem.isri import ISRIStemmer
        stemmer = ISRIStemmer()
        if not isinstance(text, str):
            return ""
        return " ".join(stemmer.stem(w) for w in text.split())
    except Exception:
        return text


def full_arabic_preprocess(text: str) -> str:
    """
    Combine all preprocessing steps into one pipeline.
    
    Args:
        text: Input Arabic text
        
    Returns:
        Fully preprocessed text
    """
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_stopwords(text)
    text = stem_arabic(text)
    return text


def get_unique_punctuation_count(text: str) -> int:
    """
    Count unique punctuation marks in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of unique punctuation characters
    """
    import regex
    if not isinstance(text, str) or not text.strip():
        return 0
    punctuations = regex.findall(r'[^\w\s]', text)
    return len(set(punctuations))


# ============================================================================
# Dataset Utilities
# ============================================================================

def flatten_dataset(dataset) -> pd.DataFrame:
    """
    Flatten the Hugging Face dataset into a unified DataFrame.
    
    Args:
        dataset: Loaded Hugging Face dataset with splits
        
    Returns:
        Pandas DataFrame with columns: abstract_text, source_split, generated_by, label
    """
    rows = []
    
    for split_name in ["by_polishing", "from_title", "from_title_and_content"]:
        split_df = dataset[split_name].to_pandas()
        
        for _, row in split_df.iterrows():
            # Human-written (label 1)
            rows.append({
                "abstract_text": row["original_abstract"],
                "source_split": split_name,
                "generated_by": "human",
                "label": 1
            })
            
            # AI-generated (label 0) from multiple models
            for model_name in ["allam", "jais", "llama", "openai"]:
                col = f"{model_name}_generated_abstract"
                if col in row and pd.notna(row[col]):
                    rows.append({
                        "abstract_text": row[col],
                        "source_split": split_name,
                        "generated_by": model_name,
                        "label": 0
                    })
    
    return pd.DataFrame(rows)


# ============================================================================
# Visualization Utilities
# ============================================================================

def save_plot(fig, filepath: str, dpi: int = 150):
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        filepath: Destination path
        dpi: Resolution for saved image
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Plot saved to: {filepath}")