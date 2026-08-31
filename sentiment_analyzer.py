import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
        # Domain-specific lexicon tuning for customer support
        self.sia.lexicon.update({
            'unresponsive': -2.5,
            'overpriced': -2.0,
            'buggy': -2.0,
            'flawless': 2.5,
            'superb': 2.5
        })

    def analyze(self, text: str) -> dict:
        """Calculates compound, positive, neutral, and negative sentiment scores."""
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']

        # Determine sentiment label
        if compound >= 0.05:
            label = 'Positive'
        elif compound <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'

        # Flag urgent negative feedback for support routing
        is_urgent = scores['neg'] >= 0.35 or compound <= -0.5

        return {
            'compound_score': round(compound, 3),
            'positive_score': round(scores['pos'], 3),
            'neutral_score': round(scores['neu'], 3),
            'negative_score': round(scores['neg'], 3),
            'label': label,
            'urgent_support_needed': is_urgent
        }