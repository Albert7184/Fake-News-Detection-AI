# 🧠 Fake News Detection AI Dashboard

An **AI-powered Fake News Detection System** with a modern analytics dashboard that analyzes news articles and predicts whether they are **Fake** or **Real** using Natural Language Processing and Machine Learning.

The system supports multiple input methods and provides interactive visualizations to help users understand prediction results.

---

# 🚀 Features

### 🧠 AI Fake News Detection

* Detect whether a news article is **Fake or Real**
* Machine Learning classification model
* Confidence score prediction
* NLP preprocessing pipeline

---

### 📰 Multiple Input Methods

Users can analyze news articles in several ways:

• **Paste article text**

• **Analyze a news URL**

• **Upload a file**

Supported file formats:

TXT
PDF
DOCX

---

### 📊 Interactive AI Dashboard

The dashboard provides:

* Prediction probability charts
* Fake vs Real analytics
* Confidence score visualization
* Interactive UI

---

### 🔍 Keyword Highlighting

The system highlights suspicious keywords often found in misleading news articles.

Example keywords:

breaking
shocking
exclusive
conspiracy
unbelievable

---

### 🌐 News Crawler

The project includes a **news crawler module** that can fetch and process real-time news content.

---

### ⚡ Fast AI Prediction

Average prediction time:

< 200ms

---

# 🧠 Machine Learning Model

The system currently uses:

**Random Forest Classifier**

Pipeline:

```
News Text
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Random Forest Model
     ↓
Prediction (Fake / Real)
```

---

# 🔬 NLP Preprocessing

Text preprocessing includes:

* Lowercasing text
* Removing punctuation
* Removing stopwords
* Normalization
* Feature extraction with TF-IDF

---

# 📊 Example Model Performance

Example confusion matrix:

```
                 Predicted
               Fake    Real

Actual Fake      920      80
Actual Real       70     930
```

Typical accuracy:

**92% – 95%**

---

# 🗂 Project Structure

```
FAKE NEWS DETECTION
│
├── data
│   └── dataset
│       ├── Fake.csv
│       └── True.csv
│
├── images
│
├── models
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── src
│   ├── data
│   │   ├── prepare_data.py
│   │   └── preprocess.py
│   │
│   ├── models
│   │   ├── evaluate.py
│   │   ├── train_bert.py
│   │   └── train_ml.py
│   │
│   ├── pipeline
│   │   ├── inference.py
│   │   └── news_crawler.py
│   │
│   └── visualization
│       └── graph_visualizer.py
│
├── static
│   ├── css
│   ├── JS
│   └── style.css
│
├── templates
│   ├── base.html
│   ├── dashboard.html
│   ├── graph.html
│   └── index.html
│
├── requirements.txt
│
├── .gitignore
├── app.py
└── README.md
```

---

# ⚙ Installation

## 1️⃣ Clone the Repository

```
git clone https://github.com/Albert7184/Fake-News-Detection-AI.git
```

Move into the project folder:

```
cd Fake-News-Detection-AI
```

---

## 2️⃣ Create Virtual Environment

Recommended:

```
python -m venv venv
```

Activate it.

Windows:

```
venv\Scripts\activate
```

Mac / Linux:

```
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

Main dependencies used:

```
Flask
scikit-learn
numpy
beautifulsoup4
requests
pdfplumber
python-docx
transformers
torch
textblob
```

---

# ▶ Running the Application

Start the Flask server:

```
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

The **Fake News AI Dashboard** will appear.

---

# 🧪 How to Use

### Method 1 — Paste News Text

1. Open the **Analyzer page**
2. Paste the article text
3. Click **Analyze**
4. View prediction results and charts

---

### Method 2 — Analyze a URL

1. Paste a news article URL
2. Click **Analyze**
3. The system extracts the article text
4. AI runs prediction automatically

---

### Method 3 — Upload a File

Supported formats:

TXT
PDF
DOCX

Steps:

1. Upload file
2. Click **Analyze**
3. AI extracts text and predicts result

Maximum file size:

5 MB

---

# 📊 Example Output

Example prediction:

```
Prediction: Fake News 🚨

Fake Probability: 83%
Real Probability: 17%
```

Charts display prediction confidence visually.

---

# ⚠ Limitations

Current limitations:

• Model accuracy depends on dataset quality
• Satirical articles may be misclassified
• Complex PDFs may not extract text perfectly
• Traditional ML models may not fully understand context

---

# 🔮 Future Improvements

Planned improvements:

* Transformer-based models (BERT / RoBERTa)
* Sentence-level fake detection
* News credibility scoring
* Source reliability analysis
* REST API version
* Real-time misinformation monitoring
* Knowledge graph for fake news

---

# 📜 License

MIT License

---

# 👨‍💻 Author

**Cao Minh Phu**
(Rowan Vale Albert)

AI / Machine Learning Developer

---

# ⭐ Support

If you find this project helpful, please consider **starring the repository on GitHub**.
