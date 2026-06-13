import argparse
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import (
    LABEL_NAMES,
    MAX_SEQUENCE_LENGTH,
    REPORTS_DIR,
    TEST_CSV_PATH,
)
from src.predict import load_deep_learning_artifacts
from src.preprocessing import preprocess_pipeline
from src.logger import logger


def evaluate_lstm_model(output_dir=REPORTS_DIR):
    """
    Evaluates the trained LSTM model on the held-out AG News test set.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading test data from {TEST_CSV_PATH}")
    df = pd.read_csv(TEST_CSV_PATH)
    processed_text = df['text'].apply(preprocess_pipeline)
    y_true = df['label'].astype('int32').values

    model, tokenizer, metadata = load_deep_learning_artifacts()
    sequences = tokenizer.texts_to_sequences(processed_text)
    X_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    probabilities = model.predict(X_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    target_names = [LABEL_NAMES[index] for index in sorted(LABEL_NAMES)]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    report_path = os.path.join(output_dir, 'classification_report.json')
    matrix_path = os.path.join(output_dir, 'confusion_matrix.csv')
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')

    with open(report_path, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    pd.DataFrame(matrix, index=target_names, columns=target_names).to_csv(matrix_path)
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(
            {
                "accuracy": float(accuracy),
                "model_type": metadata.get("model_type", "LSTM"),
                "test_rows": int(len(df)),
                "report_path": report_path,
                "confusion_matrix_path": matrix_path,
            },
            file,
            indent=2,
        )

    logger.info(f"Test accuracy: {accuracy}")
    return {
        "accuracy": float(accuracy),
        "report_path": report_path,
        "confusion_matrix_path": matrix_path,
        "summary_path": summary_path,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the trained LSTM classifier.")
    parser.add_argument('--output-dir', default=REPORTS_DIR)
    args = parser.parse_args()
    result = evaluate_lstm_model(output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
