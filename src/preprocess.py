import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

def preprocess_text(text):

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # tokenization
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)