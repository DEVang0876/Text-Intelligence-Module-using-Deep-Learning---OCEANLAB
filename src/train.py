import pandas as pd
import numpy as np
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from src.config import (
    BATCH_SIZE,
    DL_MODEL_PATH,
    EMBEDDING_DIM,
    EPOCHS,
    LABEL_NAMES,
    LSTM_UNITS,
    MAX_FEATURES,
    MAX_NUM_WORDS,
    MAX_SEQUENCE_LENGTH,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_VECTORIZER_PATH,
    TOKENIZER_PATH,
    TRAIN_CSV_PATH,
)
from src.preprocessing import preprocess_pipeline
from src.logger import logger

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

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

def prepare_texts(df):
    logger.info("Preprocessing text data...")
    df = df.copy()
    df['processed_text'] = df['text'].apply(preprocess_pipeline)
    return df

def train_baseline_model():
    """
    Trains and saves the TF-IDF + Logistic Regression baseline.
    """
    df = prepare_texts(load_data(TRAIN_CSV_PATH))
    
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
    logger.info("Baseline model and vectorizer saved successfully.")
    return {"accuracy": accuracy, "classification_report": report}

def build_lstm_model(num_classes):
    """
    Builds an improved LSTM classifier for AG News text classification.
    """
    model = Sequential([
        Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

def save_tokenizer(tokenizer):
    with open(TOKENIZER_PATH, 'w', encoding='utf-8') as file:
        file.write(tokenizer.to_json())

def save_metadata(history, validation_accuracy, class_count):
    metadata = {
        "model_type": "Bidirectional LSTM",
        "max_num_words": MAX_NUM_WORDS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "label_names": LABEL_NAMES,
        "validation_accuracy": float(validation_accuracy),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "num_classes": int(class_count),
    }
    with open(METADATA_PATH, 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)

def train_lstm_model(sample_size=None, epochs=EPOCHS):
    """
    Trains the deep learning model and saves the model, tokenizer, and metadata.
    """
    df = load_data(TRAIN_CSV_PATH)
    if sample_size:
        df = df.groupby('label', group_keys=False).sample(
            n=max(1, sample_size // df['label'].nunique()),
            random_state=RANDOM_STATE,
        )
    df = prepare_texts(df)

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['processed_text'])
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    y = df['label'].astype('int32').values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_lstm_model(num_classes=len(np.unique(y)))
    logger.info(model.summary())
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5),
        ModelCheckpoint(DL_MODEL_PATH, monitor='val_accuracy', save_best_only=True),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    validation_loss, validation_accuracy = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"LSTM validation loss: {validation_loss}")
    logger.info(f"LSTM validation accuracy: {validation_accuracy}")

    model.save(DL_MODEL_PATH)
    save_tokenizer(tokenizer)
    save_metadata(history, validation_accuracy, len(np.unique(y)))
    logger.info("LSTM model, tokenizer, and metadata saved successfully.")
    return {"validation_accuracy": float(validation_accuracy)}

def train_model(model_type='lstm', sample_size=None, epochs=EPOCHS):
    if model_type == 'baseline':
        return train_baseline_model()
    return train_lstm_model(sample_size=sample_size, epochs=epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text classification models.")
    parser.add_argument('--model-type', choices=['lstm', 'baseline'], default='lstm')
    parser.add_argument('--sample-size', type=int, default=None, help='Optional balanced sample size for quick smoke training.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()
    train_model(model_type=args.model_type, sample_size=args.sample_size, epochs=args.epochs)
