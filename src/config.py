import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test.csv')

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf.pkl')
DL_MODEL_PATH = os.path.join(MODELS_DIR, 'news_lstm.keras')
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'tokenizer.json')
METADATA_PATH = os.path.join(MODELS_DIR, 'metadata.json')

# Reports directory
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Log directory
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FILE_PATH = os.path.join(LOG_DIR, 'app.log')

# Model parameters
MAX_FEATURES = 5000
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

LABEL_NAMES = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Sci/Tech',
}

for directory in (LOG_DIR, MODELS_DIR, REPORTS_DIR):
    os.makedirs(directory, exist_ok=True)
