import json
import os
import joblib
import numpy as np
from src.config import (
    DL_MODEL_PATH,
    LABEL_NAMES,
    MAX_SEQUENCE_LENGTH,
    METADATA_PATH,
    MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    TOKENIZER_PATH,
)
from src.preprocessing import preprocess_pipeline
from src.logger import logger

def load_baseline_model_and_vectorizer():
    """
    Loads the trained model and TF-IDF vectorizer.
    
    Returns:
        tuple: The loaded model and vectorizer.
    """
    logger.info("Loading the model and TF-IDF vectorizer.")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    return model, vectorizer

def load_deep_learning_artifacts():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    logger.info("Loading the LSTM model and tokenizer.")
    model = tf.keras.models.load_model(DL_MODEL_PATH)
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as file:
        tokenizer = tokenizer_from_json(file.read())
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
    return model, tokenizer, metadata

def label_to_name(label):
    return LABEL_NAMES.get(int(label), str(label))

def predict_with_deep_learning(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer, metadata = load_deep_learning_artifacts()
    processed_text = preprocess_pipeline(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    probabilities = model.predict(padded, verbose=0)[0]
    label = int(np.argmax(probabilities))
    return {
        "label": label,
        "category": label_to_name(label),
        "confidence": float(probabilities[label]),
        "probabilities": {
            label_to_name(index): float(probability)
            for index, probability in enumerate(probabilities)
        },
        "model_type": metadata.get("model_type", "LSTM"),
    }

def predict_with_baseline(text):
    model, vectorizer = load_baseline_model_and_vectorizer()
    processed_text = preprocess_pipeline(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = int(model.predict(vectorized_text)[0])
    result = {
        "label": prediction,
        "category": label_to_name(prediction),
        "confidence": None,
        "probabilities": {},
        "model_type": "TF-IDF Logistic Regression",
    }
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(vectorized_text)[0]
        result["confidence"] = float(np.max(probabilities))
        result["probabilities"] = {
            label_to_name(index): float(probability)
            for index, probability in enumerate(probabilities)
        }
    return result

def predict_text_details(text):
    """
    Predicts the category for a given text and returns rich prediction details.
    
    Args:
        text (str): The input text.
        
        dict: Prediction label, category, confidence, probabilities, and model type.
    """
    logger.info(f"Predicting for text: '{text[:50]}...'")
    if os.path.exists(DL_MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        result = predict_with_deep_learning(text)
    elif os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_VECTORIZER_PATH):
        result = predict_with_baseline(text)
    else:
        raise FileNotFoundError(
            "No trained model artifacts found. Run `python -m src.train --model-type lstm` first."
        )
    logger.info(f"Prediction: {result['category']} ({result['label']})")
    return result

def predict_text(text):
    """
    Backwards-compatible helper that returns only the numeric label.
    """
    return predict_text_details(text)["label"]

if __name__ == "__main__":
    sample_text = "This is a sample text to test the prediction."
    prediction = predict_text_details(sample_text)
    print(f"The predicted category for the text is: {prediction['category']}")
