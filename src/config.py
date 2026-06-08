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

# Log directory
LOG_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE_PATH = os.path.join(LOG_DIR, 'app.log')

# Model parameters
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42
