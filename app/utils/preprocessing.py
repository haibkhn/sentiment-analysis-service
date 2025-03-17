import re

def preprocess_text(text):
    """
    Perform basic text preprocessing on review text.
    
    Args:
        text (str): The raw text to preprocess
        
    Returns:
        str: Preprocessed text ready for sentiment analysis
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags if present (for website reviews)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters but keep punctuation that might be relevant for sentiment
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_review_batch(reviews):
    """
    Preprocess a batch of reviews.
    
    Args:
        reviews (list): List of review texts
        
    Returns:
        list: List of preprocessed review texts
    """
    return [preprocess_text(review) for review in reviews]