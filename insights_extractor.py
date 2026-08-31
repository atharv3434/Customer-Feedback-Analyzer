from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor import TextPreprocessor


class InsightsExtractor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def extract_top_keywords(self, corpus: list[str], top_n: int = 5) -> list[tuple[str, float]]:
        """Extracts top N representative keywords across the feedback corpus."""
        # Tokenize and lemmatize every document
        tokenized_corpus = [
            " ".join(self.preprocessor.tokenize_and_lemmatize(doc))
            for doc in corpus
        ]

        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(tokenized_corpus)

        # Average TF-IDF weights across all documents
        mean_scores = tfidf_matrix.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()

        ranked_terms = sorted(zip(terms, mean_scores), key=lambda x: x[1], reverse=True)
        return ranked_terms[:top_n]