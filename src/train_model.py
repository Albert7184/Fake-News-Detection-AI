import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# Load dataset
data = pd.read_csv("dataset/fake_news.csv")

X = data["text"]
y = data["label"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# Create pipeline
pipeline = Pipeline([
("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
("model", RandomForestClassifier(n_estimators=200))
])


# Train model
pipeline.fit(X_train,y_train)


# Save model
pickle.dump(pipeline, open("models/fake_news_model.pkl","wb"))

print("Model trained and saved!")