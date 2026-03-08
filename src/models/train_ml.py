import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from models.data.preprocess import preprocess_text

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

print("\nLoading dataset...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fake_path = os.path.join(BASE_DIR, "data", "dataset", "Fake.csv")
true_path = os.path.join(BASE_DIR, "data", "dataset", "True.csv")

fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", data.shape)

data = data[["text", "label"]]


# =========================================================
# 2. PREPROCESS TEXT
# =========================================================

print("\nCleaning text...")

data["clean_text"] = data["text"].apply(preprocess_text)


# =========================================================
# 3. TF-IDF VECTORIZATION
# =========================================================

print("\nVectorizing text (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

print("TF-IDF shape:", X.shape)


# =========================================================
# 4. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])


# =========================================================
# 5. DEFINE MODELS
# =========================================================

models = {

    "Logistic Regression": LogisticRegression(max_iter=2000),

    "Naive Bayes": MultinomialNB(),

    "SVM": LinearSVC(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1
    )

}


# =========================================================
# 6. TRAIN MODELS
# =========================================================

results = {}

best_model = None
best_accuracy = 0
best_model_name = ""
best_predictions = None

print("\nTraining models...")

for name, model in models.items():

    print("\n==============================")
    print("Training:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    results[name] = accuracy

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
        best_predictions = y_pred


# =========================================================
# 7. MODEL LEADERBOARD
# =========================================================

print("\n==============================")
print("MODEL LEADERBOARD")

leaderboard = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

leaderboard = leaderboard.sort_values(by="Accuracy", ascending=False)

print(leaderboard)


# =========================================================
# 8. MODEL COMPARISON CHART
# =========================================================

print("\nDrawing model comparison chart...")

plt.figure(figsize=(8,5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=leaderboard
)

plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.tight_layout()

plt.show()


# =========================================================
# 9. BEST MODEL
# =========================================================

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)


# =========================================================
# 10. SAVE MODEL
# =========================================================

print("\nSaving model...")

models_dir = os.path.join(BASE_DIR, "models")

os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "fake_news_model.pkl")
vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")

pickle.dump(best_model, open(model_path, "wb"))
pickle.dump(vectorizer, open(vectorizer_path, "wb"))

print("Model saved in models/")


# =========================================================
# 11. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6,4))

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


# =========================================================
# 12. FEATURE IMPORTANCE (TOP WORDS)
# =========================================================

print("\nTop words influencing prediction...")

feature_names = vectorizer.get_feature_names_out()

try:

    if hasattr(best_model, "coef_"):

        coef = best_model.coef_[0]

        top_fake_idx = coef.argsort()[:20]
        top_real_idx = coef.argsort()[-20:]

        print("\nTop words for FAKE news:")
        print([feature_names[i] for i in top_fake_idx])

        print("\nTop words for REAL news:")
        print([feature_names[i] for i in top_real_idx])

    else:
        print("Feature importance not available for this model.")

except:
    print("Could not compute feature importance.")