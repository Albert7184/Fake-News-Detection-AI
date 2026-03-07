import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess_text

# ML tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# 1. LOAD DATASET
# =========================================================

data = pd.read_csv("data/news.csv")

print("Dataset size:", data.shape)


# =========================================================
# 2. TEXT PREPROCESSING
# =========================================================

print("Cleaning text...")

data["clean_text"] = data["text"].apply(preprocess_text)


# =========================================================
# 3. TEXT -> NUMBERS (TF-IDF)
# =========================================================

vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2)   # unigram + bigram
)

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

print("TF-IDF shape:", X.shape)


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])


# =========================================================
# 5. DEFINE MODELS
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}


best_model = None
best_accuracy = 0
best_model_name = ""
best_predictions = None


# =========================================================
# 6. TRAIN + EVALUATE MODELS
# =========================================================

for name, model in models.items():

    print("\n=================================")
    print("Training:", name)

    # train
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # confusion matrix
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # check best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
        best_predictions = y_pred


# =========================================================
# 7. BEST MODEL RESULT
# =========================================================

print("\n=================================")
print("Best Model:", best_model_name)
print("Best Accuracy:", best_accuracy)


# =========================================================
# 8. SAVE BEST MODEL
# =========================================================

pickle.dump(best_model, open("models/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Best model saved in models/")


# =========================================================
# 9. VISUALIZE CONFUSION MATRIX (BEST MODEL)
# =========================================================

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(f"Confusion Matrix ({best_model_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()