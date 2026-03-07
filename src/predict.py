import pickle
from preprocess import preprocess_text

# Load model
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Test article
news = input("Enter news text: ")

# Preprocess
clean_text = preprocess_text(news)

# Vectorize
vector = vectorizer.transform([clean_text])

# Predict
prediction = model.predict(vector)

if prediction[0] == 0:
    print("Fake News")
else:
    print("Real News")