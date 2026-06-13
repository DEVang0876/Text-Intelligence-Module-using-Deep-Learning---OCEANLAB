import html
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    """
    Downloads necessary NLTK resources if they are not already present.
    """
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "wordnet": "corpora/wordnet",
        "stopwords": "corpora/stopwords",
    }
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if os.getenv('ALLOW_NLTK_DOWNLOADS') == '1':
                try:
                    nltk.download(resource_name, quiet=True)
                except Exception:
                    pass

# Call this once to ensure resources are available
download_nltk_resources()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with',
    }

try:
    nltk.data.find('corpora/wordnet')
    WORDNET_AVAILABLE = True
except LookupError:
    WORDNET_AVAILABLE = False

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
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'\\[a-z]+', ' ', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', ' ', text)
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

    tokens = re.findall(r'[a-z]+', text)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) if WORDNET_AVAILABLE else word
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]
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
