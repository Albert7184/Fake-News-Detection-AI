# 🧠 Fake News Detection AI

A Machine Learning web application that detects whether a news article is **Fake** or **Real** using Natural Language Processing and a **Random Forest classifier**.

This project provides a simple web interface where users can analyze news articles by:

* 📝 Pasting article text
* 🌐 Providing a news URL
* 📂 Uploading a file (TXT, PDF, DOCX)

The system processes the text, applies NLP preprocessing, and predicts the probability that the article is **Fake or Real**.

---

# 🚀 Features

* Fake news detection using Machine Learning
* NLP preprocessing pipeline
* Upload files (TXT, PDF, DOCX)
* Extract article text from URLs
* Highlight suspicious keywords in the article
* AI confidence score visualization using Chart.js
* File upload validation and error handling
* Modern interactive UI

---

# 🧠 Machine Learning Model

The model used in this project is:

**Random Forest Classifier**

Machine Learning Pipeline:

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

### Text preprocessing includes

* Lowercasing
* Removing punctuation
* Removing stopwords
* Lemmatization / normalization

---

# 📊 Model Evaluation

Example confusion matrix of the trained model:

```
                 Predicted
               Fake    Real

Actual Fake      920      80
Actual Real       70     930
```

Typical accuracy achieved:

```
~92% – 95%
```

You can optionally include a confusion matrix image:

```
images/confusion_matrix.png
```

Add it to README like this:

```
![Confusion Matrix](images/confusion_matrix.png)
```

---

# 🗂 Project Structure

```
fake-news-detector
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── src
│   ├── preprocess.py
│   └── predict.py
│
├── static
│   └── style.css
│
├── templates
│   └── index.html
│
├── notebooks
│   └── model_training.ipynb
│
└── images
    └── image.png
```

---

# ⚙ Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/fake-news-detector.git

cd fake-news-detector
```

---

### 2. Create a virtual environment (recommended)

```
python -m venv venv
```

Activate it:

Windows

```
venv\Scripts\activate
```

Mac / Linux

```
source venv/bin/activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

Example dependencies used:

```
flask
scikit-learn
pandas
numpy
beautifulsoup4
requests
pdfplumber
python-docx
```

---

# ▶ Running the Application

Run the Flask server:

```
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

# 🧪 How to Use

### Method 1 — Paste News Text

1. Paste a news article into the text box
2. Click **Analyze with AI**
3. View prediction and probability scores

---

### Method 2 — Analyze News URL

1. Paste a news article link into the URL field
2. Click **Analyze with AI**
3. The system extracts the article content and analyzes it

---

### Method 3 — Upload News File

Supported file formats:

```
TXT
PDF
DOCX
```

Steps:

1. Click **Upload file**
2. Select a supported file
3. Click **Analyze with AI**

Maximum upload size:

```
5 MB
```

---

# 🎯 Example Output

```
Prediction: Fake News 🚨

Fake Probability: 83%
Real Probability: 17%
```

A chart visualization displays the confidence score.

---

# 🔍 Fake Keyword Highlighting

The system highlights certain suspicious keywords that frequently appear in misleading or sensational news articles:

```
breaking
shocking
exclusive
conspiracy
unbelievable
```

These words are visually marked in the analyzed text.

---

# ⚠ Limitations

* Model accuracy depends on dataset quality
* Satirical articles may be misclassified
* Some complex PDF layouts may not extract text perfectly

---

# 🔮 Future Improvements

Possible future upgrades:

* Use transformer models (BERT / RoBERTa)
* Sentence-level fake detection
* Credibility scoring system
* News source reliability analysis
* REST API version of the detector
* Real-time news verification

---

# 📜 License

MIT License

---

# 👨‍💻 Author

Cao Minh Phu (Rowan Vale Albert)
