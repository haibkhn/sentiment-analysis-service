from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        Uses DistilBERT model fine-tuned for sentiment analysis.
        """
        # Load pre-trained model for sentiment analysis
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    
    def analyze(self, text):
        """
        Analyze the sentiment of the provided text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Dictionary containing sentiment analysis results
        """
        # Handle long texts - transformer models typically have a 512 token limit
        # If text is too long, truncate it to approximately 250 words as sometimes we can already get a good sense of sentiment
        # This is a simple approach - in production, we might want more sophisticated chunking
        words = text.split()
        if len(words) > 250:  # More conservative truncation
            text = " ".join(words[:250])
        
        # Get raw sentiment analysis results
        result = self.sentiment_pipeline(text)[0]
        
        # Extract label and score
        label = result['label']
        score = result['score']
        
        # Convert to positive/negative format
        sentiment = "positive" if label == "POSITIVE" else "negative"
        
        # Create normalized score between -1 and 1
        # Where 1 is very positive and -1 is very negative
        normalized_score = score if sentiment == "positive" else -score
        
        return {
            "sentiment": sentiment,
            "confidence": score,
            "normalized_score": normalized_score,
            "detailed_scores": {
                "POSITIVE": score if sentiment == "positive" else 1 - score,
                "NEGATIVE": score if sentiment == "negative" else 1 - score
            }
        }