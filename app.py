from preprocessor import TextPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from insights_extractor import InsightsExtractor


def main():
    # Sample real-world dataset
    sample_reviews = [
        "The software is completely unresponsive and buggy. It crashed three times during my presentation!",
        "Absolutely superb customer service! The team resolved my billing issue in less than five minutes.",
        "The interface is decent, but the annual subscription is overpriced for the features offered.",
        "Seamless setup and flawless integration with my existing database. Highly recommended!",
        "Average tool. It does the job, but the dashboard loading speed is quite slow."
    ]

    print("=" * 80)
    print(" " * 25 + "FEEDBACK ANALYSIS REPORT")
    print("=" * 80)

    preprocessor = TextPreprocessor()
    sentiment_engine = SentimentAnalyzer()
    insights_engine = InsightsExtractor()

    # 1. Document-level sentiment & urgency analysis
    for idx, review in enumerate(sample_reviews, 1):
        clean_tokens = preprocessor.tokenize_and_lemmatize(review)
        sentiment = sentiment_engine.analyze(review)

        print(f"\n[Feedback #{idx}]")
        print(f"Raw Text : {review}")
        print(f"Tokens   : {clean_tokens}")
        print(f"Sentiment: {sentiment['label']} (Compound: {sentiment['compound_score']})")
        print(f"Urgent   : {'⚠️ YES - Route to Support' if sentiment['urgent_support_needed'] else 'No'}")
        print("-" * 80)

    # 2. Corpus-level keyword insights
    top_keywords = insights_engine.extract_top_keywords(sample_reviews, top_n=5)
    print("\n" + "=" * 80)
    print(" " * 28 + "TOP CORPUS KEYWORDS")
    print("=" * 80)
    for rank, (word, score) in enumerate(top_keywords, 1):
        print(f"{rank}. {word:<20} (TF-IDF Weight: {score:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()