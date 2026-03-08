import re
import nltk
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# =====================================================
# DOWNLOAD NLTK DATA (SAFE)
# =====================================================

def download_nltk():

    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4"
    }

    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


download_nltk()


# =====================================================
# NLP TOOLS
# =====================================================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

punct_table = str.maketrans("", "", string.punctuation)


# =====================================================
# BASIC CLEANING
# =====================================================

def clean_basic(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove html
    text = re.sub(r"<.*?>", " ", text)

    # remove mentions
    text = re.sub(r"@\w+", " ", text)

    # remove numbers
    text = re.sub(r"\d+", " ", text)

    # remove punctuation
    text = text.translate(punct_table)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# FULL PREPROCESS PIPELINE
# =====================================================

def preprocess_text(text):

    text = clean_basic(text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)