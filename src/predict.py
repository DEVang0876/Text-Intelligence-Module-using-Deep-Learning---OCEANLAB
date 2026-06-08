import joblib
from src.config import MODEL_PATH, TFIDF_VECTORIZER_PATH
from src.preprocessing import preprocess_pipeline
from src.logger import logger

def load_model_and_vectorizer():
    """
    Loads the trained model and TF-IDF vectorizer.
    
    Returns:
        tuple: The loaded model and vectorizer.
    """
    logger.info("Loading the model and TF-IDF vectorizer.")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    return model, vectorizer

def predict_text(text):
    """

    Predicts the label for a given text.
    
    Args:
        text (str): The input text.
        
    Returns:
        str: The predicted label.
    """
    logger.info(f"Predicting for text: '{text[:50]}...'")
    model, vectorizer = load_model_and_vectorizer()
    
    processed_text = preprocess_pipeline(text)
    vectorized_text = vectorizer.transform([processed_text])
    
    prediction = model.predict(vectorized_text)
    
    logger.info(f"Prediction: {prediction[0]}")
    return prediction[0]

if __name__ == "__main__":
    sample_text = "This is a sample text to test the prediction."
    prediction = predict_text(sample_text)
    print(f"The predicted label for the text is: {prediction}")
