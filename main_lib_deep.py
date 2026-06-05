import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

important_words = {
    "up", "down", "buy", "sell", "bull", "bear", "bullish", "bearish",
    "profit", "loss", "long", "short", "gain", "gains", "drop", "drops",
    "rally", "rallies", "rise", "rises", "fall", "falls", "low", "high",
    "beat", "miss", "upgrade", "downgrade"
}

stop_words = set(ENGLISH_STOP_WORDS) - important_words

def clean_text(text):
    """Clean a tweet using regular expressions."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)       # remove URLs
    text = re.sub(r"@\w+", " ", text)                            # remove mentions
    text = re.sub(r"#", "", text)                                 # keep hashtag words
    text = re.sub(r"\$([a-zA-Z]+)", r" \1 ", text)               # $AAPL -> AAPL
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                     # remove symbols/numbers
    text = re.sub(r"\s+", " ", text).strip()                     # remove extra spaces
    return text


# Stemming and Lemmatization
lemma = WordNetLemmatizer()
def stem_simple(word):
    """A small fallback stemmer so the notebook runs without extra downloads."""
    for suffix in ['ing','edly','ed','ly','ies','s']:
        if word.endswith(suffix) and len(word) > len(suffix)+3:
            if suffix == "ies":
                return word[:-3] + "y"
            return word[:-len(suffix)]
    return word

def preprocess_lemma(text):
    """Basic cleaning + stopword removal. Used as the main normalized text column."""
    cleaned = clean_text(text)
    tokens = [lemma.lemmatize(w) for w in cleaned.split() if w not in stop_words and len(w)>2]
    return " ".join(tokens)

def preprocess_stem(text):
    cleaned = clean_text(text)
    tokens = [stem_simple(w) for w in cleaned.split() if w not in stop_words and len(w)>2]
    return " ".join(tokens)


# =========================
# Deep Learning utilities
# =========================
# These helpers are intentionally optional: the classic sklearn pipeline still
# runs even when TensorFlow / PyTorch / Transformers are not installed.

import os
import random
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

LABEL_NAMES = ["Bearish", "Bullish", "Neutral"]
LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


def set_global_seed(seed=42):
    """Set random seeds for reproducible sklearn / numpy / TF / Torch experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def evaluate_classification(y_true, y_pred, model_name, elapsed_seconds=None):
    """Return a dict with the same metrics used throughout the notebook."""
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    row = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    }
    if elapsed_seconds is not None:
        row["Total_evaluation_time"] = elapsed_seconds
    return row


def print_classification_summary(y_true, y_pred):
    """Print a readable classification report using the project label names."""
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        zero_division=0,
    ))


def make_submission(test_ids, predictions, output_path="submission.csv"):
    """Create the Kaggle/class submission file with columns id,label."""
    submission = pd.DataFrame({"id": test_ids, "label": predictions.astype(int)})
    submission.to_csv(output_path, index=False)
    return submission
