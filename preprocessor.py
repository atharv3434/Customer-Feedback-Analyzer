import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data bundles
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Removes URLs, punctuation, numbers, and excess whitespace."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Lowercase
        text = text.lower()
        # Remove numbers and punctuation
        text = re.sub(r'[\d' + re.escape(string.punctuation) + r']', ' ', text)
        # Strip extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> list[str]:
        """Cleans, tokenizes, removes stopwords, and lemmatizes tokens."""
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned)
        
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return processed_tokens