"""
Fake News Detection AI
Full Dashboard Backend
Flask + AI Model
"""

from flask import Flask, render_template, request, jsonify

import pickle
import feedparser
import requests

from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from langdetect import detect
from textblob import TextBlob

from src.data.preprocess import preprocess_text


# ======================================
# INIT APP
# ======================================

app = Flask(__name__)


# ======================================
# PATHS
# ======================================

MODEL_PATH = Path("models/fake_news_model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")


# ======================================
# GLOBAL ANALYTICS
# ======================================

analytics = {
    "total": 0,
    "fake": 0,
    "real": 0
}

history = []
trend = []


# ======================================
# LOAD AI MODEL
# ======================================

print("Loading AI model...")

try:

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    print("Model loaded successfully")

except Exception as e:

    print("Model load error:", e)

    model = None
    vectorizer = None


# ======================================
# RSS NEWS SOURCES
# ======================================

news_feeds = [

    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://rss.cnn.com/rss/edition.rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"

]


# ======================================
# NEWS CRAWLER
# ======================================

def crawl_news(limit=5):

    articles = []

    for url in news_feeds:

        try:

            feed = feedparser.parse(url)

            for entry in feed.entries[:limit]:

                articles.append({

                    "title": entry.get("title"),
                    "link": entry.get("link"),
                    "source": feed.feed.get("title", "News")

                })

        except Exception:
            continue

    return articles


# ======================================
# SENTIMENT ANALYSIS
# ======================================

def analyze_sentiment(text):

    try:
        polarity = TextBlob(text).sentiment.polarity
    except Exception:
        polarity = 0

    if polarity > 0.3:
        return "Positive"

    if polarity < -0.3:
        return "Negative"

    return "Neutral"


# ======================================
# EXTRACT ARTICLE FROM URL
# ======================================

def extract_article(url):

    try:

        page = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        soup = BeautifulSoup(page.text, "html.parser")

        paragraphs = soup.find_all("p")

        article = " ".join(p.get_text() for p in paragraphs)

        return article[:3000]

    except Exception:

        return None


# ======================================
# AI EXPLANATION
# ======================================

def explain_prediction(text):

    suspicious = [

        "shocking",
        "breaking",
        "secret",
        "conspiracy",
        "miracle",
        "100%"

    ]

    words = text.lower().split()

    hits = [w for w in words if w in suspicious]

    if hits:

        return "Suspicious keywords detected: " + ", ".join(set(hits))

    return "No suspicious keywords detected"


# ======================================
# HOME PAGE
# ======================================

@app.route("/")
def home():

    return render_template("index.html")


# ======================================
# LIVE NEWS API
# ======================================

@app.route("/api/news")
def api_news():

    return jsonify({

        "articles": crawl_news()

    })


# ======================================
# ANALYTICS API
# ======================================

@app.route("/api/analytics")
def api_analytics():

    total = analytics["total"]

    fake_percent = 0
    real_percent = 0

    if total > 0:

        fake_percent = round((analytics["fake"] / total) * 100, 2)
        real_percent = round((analytics["real"] / total) * 100, 2)

    return jsonify({

        "total_predictions": total,
        "fake_news": analytics["fake"],
        "real_news": analytics["real"],
        "fake_percent": fake_percent,
        "real_percent": real_percent

    })


# ======================================
# HISTORY API
# ======================================

@app.route("/api/history")
def api_history():

    return jsonify({

        "history": history[-20:]

    })


# ======================================
# TREND API
# ======================================

@app.route("/api/trend")
def api_trend():

    return jsonify({

        "trend": trend[-30:]

    })


# ======================================
# SERVER STATUS
# ======================================

@app.route("/api/status")
def api_status():

    return jsonify({

        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "total_predictions": analytics["total"]

    })


# ======================================
# AI PREDICTION
# ======================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:

        return jsonify({"error": "Invalid request"}), 400

    text = data.get("news", "")
    url = data.get("url", "")

    if not text and url:
        text = extract_article(url)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:

        clean = preprocess_text(text)

        vector = vectorizer.transform([clean])

        probs = model.predict_proba(vector)[0]

        pred = model.predict(vector)[0]

    except Exception as e:

        return jsonify({"error": str(e)}), 500


    fake_prob = round(probs[0] * 100, 2)
    real_prob = round(probs[1] * 100, 2)

    result = "FAKE" if pred == 0 else "REAL"

    confidence = max(fake_prob, real_prob)

    language = detect(text)

    sentiment = analyze_sentiment(text)

    explanation = explain_prediction(text)


    # UPDATE ANALYTICS

    analytics["total"] += 1

    if pred == 0:
        analytics["fake"] += 1
    else:
        analytics["real"] += 1


    timestamp = datetime.now().strftime("%H:%M:%S")


    history.append({

        "time": timestamp,
        "result": result,
        "text": text[:120]

    })


    trend.append({

        "time": timestamp,
        "fake": analytics["fake"],
        "real": analytics["real"]

    })


    return jsonify({

        "prediction": result,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "language": language,
        "sentiment": sentiment,
        "explanation": explanation

    })


# ======================================
# RUN SERVER
# ======================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=10000,
        debug=True
    )