"""
Fake News Detection Web App
Author: Rowan Vale Albert
"""

from flask import Flask, render_template, request
import pickle
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from src.preprocess import preprocess_text

from docx import Document
import pdfplumber


# =========================================================
# App Configuration
# =========================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB


@app.errorhandler(413)
def file_too_large(e):

    return render_template(
        "index.html",
        prediction="❌ Upload failed: File is too large (max 5MB)"
    )


MODEL_PATH = Path("models/fake_news_model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")

history = []

fake_keywords = [
    "shocking",
    "conspiracy",
    "secret",
    "unbelievable",
    "breaking",
    "exclusive"
]


# =========================================================
# Load Model
# =========================================================

def load_models():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_models()


# =========================================================
# Extract article from URL
# =========================================================

def extract_article(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        page = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(page.text, "html.parser")

        paragraphs = soup.find_all("p")

        article = " ".join(p.text for p in paragraphs)

        return article

    except Exception:
        return None


# =========================================================
# Read uploaded file
# =========================================================

def read_uploaded_file(file):

    filename = file.filename.lower()

    try:

        if filename.endswith(".txt"):

            return file.read().decode("utf-8", errors="ignore")

        elif filename.endswith(".docx"):

            doc = Document(file)

            return "\n".join(p.text for p in doc.paragraphs)

        elif filename.endswith(".pdf"):

            text = ""

            with pdfplumber.open(file) as pdf:

                for page in pdf.pages[:5]:

                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"

            return text.strip()

        else:
            return None

    except Exception as e:

        print("File read error:", e)

        return None


# =========================================================
# Highlight fake keywords
# =========================================================

def highlight_fake(text):

    for word in fake_keywords:

        text = text.replace(
            word,
            f'<span class="fake-highlight">{word}</span>'
        )

    return text


# =========================================================
# Routes
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        history=history[-5:]
    )


@app.route("/predict", methods=["POST"])
def predict():

    news_text = request.form.get("news", "").strip()
    url = request.form.get("url", "").strip()
    file = request.files.get("file")

    upload_message = None

    # =============================
    # FILE UPLOAD
    # =============================

    if file and file.filename != "":

        filename = file.filename.lower()

        if not filename.endswith((".txt", ".pdf", ".docx")):

            return render_template(
                "index.html",
                prediction="❌ Upload failed: Unsupported file type"
            )

        content = read_uploaded_file(file)

        if not content:

            return render_template(
                "index.html",
                prediction="❌ Upload failed: Could not read file"
            )

        news_text = content
        upload_message = f"✅ Upload successful: {filename}"

    # =============================
    # URL CRAWL
    # =============================

    if url and not news_text:

        article = extract_article(url)

        if article:
            news_text = article
        else:
            return render_template(
                "index.html",
                prediction="❌ Could not extract article"
            )

    if not news_text:

        return render_template(
            "index.html",
            prediction="⚠️ Please enter news text"
        )

    # =============================
    # AI Prediction
    # =============================

    clean_text = preprocess_text(news_text)

    vector = vectorizer.transform([clean_text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    fake_prob = round(probabilities[0] * 100, 2)
    real_prob = round(probabilities[1] * 100, 2)

    result = "Fake News 🚨" if prediction == 0 else "Real News ✅"

    highlighted_text = highlight_fake(news_text)

    history.append({
        "text": news_text[:120],
        "result": result
    })

    return render_template(
        "index.html",
        prediction=result,
        fake_prob=fake_prob,
        real_prob=real_prob,
        news_text=news_text,
        highlighted_text=highlighted_text,
        upload_message=upload_message,
        history=history[-5:]
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)