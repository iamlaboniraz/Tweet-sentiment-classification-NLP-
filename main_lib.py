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



