import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_resources():
    """
    Downloads necessary NLTK resources if they are not already present.
    """
    resources = {
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
        "stopwords": "corpora/stopwords"
    }
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except nltk.downloader.DownloadError:
            nltk.download(resource_name)

# Call this once to ensure resources are available
download_nltk_resources()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans text by converting to lowercase, removing punctuation, and stripping extra whitespace.
    
    Args:
        text (str): The input text.
        
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    """
    Tokenizes, lemmatizes, and removes stopwords from the text.
    
    Args:
        text (str): The input text.
        
    Returns:
        str: The processed text as a single string.
    """

    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmatized_tokens)

def preprocess_pipeline(text):
    """
    A full preprocessing pipeline that cleans and lemmatizes text.
    
    Args:
        text (str): The raw input text.
        
    Returns:
        str: The fully preprocessed text.
    """
    cleaned_text = clean_text(text)
    processed_text = lemmatize_text(cleaned_text)
    return processed_text

