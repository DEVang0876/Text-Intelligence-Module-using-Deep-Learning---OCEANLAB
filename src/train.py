import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.config import TRAIN_CSV_PATH, MODEL_PATH, TFIDF_VECTORIZER_PATH, MAX_FEATURES, TEST_SIZE, RANDOM_STATE
from src.preprocessing import preprocess_pipeline
from src.logger import logger

def load_data(file_path):
    """
    Loads data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded data.
    """
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def train_model():
    """
    Trains the model and saves it along with the TF-IDF vectorizer.
    """
    df = load_data(TRAIN_CSV_PATH)
    
    # Preprocess the text data
    logger.info("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_pipeline)
    
    # Feature Engineering with TF-IDF
    logger.info("Starting feature engineering with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    X = tfidf.fit_transform(df['processed_text'])
    y = df['label']
    
    # Split data
    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Model Training
    logger.info("Training the Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Model Evaluation
    logger.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Model Accuracy: {accuracy}")
    logger.info(f"Classification Report:\n{report}")
    
    # Save the model and vectorizer
    logger.info("Saving the model and TF-IDF vectorizer...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, TFIDF_VECTORIZER_PATH)
    logger.info("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    train_model()
