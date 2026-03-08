from deep_translator import GoogleTranslator
from langdetect import detect
from models.data.preprocess import preprocess_text


def predict_news(text, model, vectorizer):

    if not text or text.strip() == "":
        return None, 0

    try:
        lang = detect(text)

        if lang != "en":
            text = GoogleTranslator(source="auto", target="en").translate(text)

    except:
        pass

    text = text[:2000]

    clean = preprocess_text(text)

    if not clean:
        return None, 0

    vector = vectorizer.transform([clean])

    prediction = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):

        prob = model.predict_proba(vector)[0]

        confidence = max(prob)

    else:

        confidence = 0.9

    return prediction, round(confidence, 3)