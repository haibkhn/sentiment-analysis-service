"""
Script to download the Hugging Face model used for sentiment analysis.
Run this script before starting the application to avoid download delays on first request.
"""

import os
import sys
from pathlib import Path

def download_model():
    """
    Download the sentiment analysis model (distilbert-base-uncased-finetuned-sst-2-english)
    to the default cache location for Hugging Face models.
    """
    print("Downloading sentiment analysis model...")
    
    # Import here to ensure transformers is installed
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Model name
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Download model and tokenizer
    print(f"Downloading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Model downloaded successfully!")
    
    # Verify with a quick test
    print("Testing model with a sample text...")
    from transformers import pipeline
    
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    test_text = "This product is perfect."
    result = sentiment_pipeline(test_text)[0]
    
    print(f"Test result: {result}")
    print("\nModel is ready for use!")

if __name__ == "__main__":
    download_model()