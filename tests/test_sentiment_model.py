import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.sentiment import SentimentAnalyzer

# Initialize model once for all tests to avoid reloading for each test
@pytest.fixture(scope="module")
def analyzer():
    """Initialize and return the sentiment analyzer"""
    return SentimentAnalyzer()

# Test positive reviews
@pytest.mark.parametrize("text", [
    "This product is amazing! I love it so much.",
    "Great customer service, very helpful and responsive.",
    "The quality exceeded my expectations. Would definitely buy again."
])
def test_positive_sentiment(analyzer, text):
    """Test that positive texts are correctly identified"""
    result = analyzer.analyze(text)
    
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.5
    assert result["normalized_score"] > 0
    assert "POSITIVE" in result["detailed_scores"]
    assert "NEGATIVE" in result["detailed_scores"]
    assert result["detailed_scores"]["POSITIVE"] > result["detailed_scores"]["NEGATIVE"]

# Test negative reviews
@pytest.mark.parametrize("text", [
    "Terrible experience. I regret buying this product.",
    "The customer support was unhelpful and rude.",
    "Poor quality, broke after one week. Waste of money."
])
def test_negative_sentiment(analyzer, text):
    """Test that negative texts are correctly identified"""
    result = analyzer.analyze(text)
    
    assert result["sentiment"] == "negative"
    assert result["confidence"] > 0.5
    assert result["normalized_score"] < 0
    assert "POSITIVE" in result["detailed_scores"]
    assert "NEGATIVE" in result["detailed_scores"]
    assert result["detailed_scores"]["NEGATIVE"] > result["detailed_scores"]["POSITIVE"]

def test_output_format(analyzer):
    """Test that the output has the correct format"""
    text = "Test review for format verification."
    result = analyzer.analyze(text)
    
    # Check that all expected keys are present
    assert "sentiment" in result
    assert "confidence" in result
    assert "normalized_score" in result
    assert "detailed_scores" in result
    
    # Check types
    assert isinstance(result["sentiment"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["normalized_score"], float)
    assert isinstance(result["detailed_scores"], dict)
    
    # Confidence should be between 0 and 1
    assert 0 <= result["confidence"] <= 1
    
    # Normalized score should be between -1 and 1
    assert -1 <= result["normalized_score"] <= 1