import pytest
from fastapi.testclient import TestClient
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# Create test client
client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "docs_url" in data

def test_analyze_positive_sentiment():
    """Test sentiment analysis with positive text"""
    review = {
        "text": "This product is amazing! I absolutely love it.",
        "review_id": "test123",
        "source": "website"
    }
    response = client.post("/api/v1/analyze", json=review)
    assert response.status_code == 200
    result = response.json()
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.5
    assert result["normalized_score"] > 0
    assert result["review_id"] == "test123"
    assert result["source"] == "website"
    assert "POSITIVE" in result["detailed_scores"]
    assert "NEGATIVE" in result["detailed_scores"]

def test_analyze_negative_sentiment():
    """Test sentiment analysis with negative text"""
    review = {
        "text": "This is terrible. I regret buying this product.",
        "review_id": "test456",
        "source": "app"
    }
    response = client.post("/api/v1/analyze", json=review)
    assert response.status_code == 200
    result = response.json()
    assert result["sentiment"] == "negative"
    assert result["confidence"] > 0.5
    assert result["normalized_score"] < 0
    assert result["review_id"] == "test456"
    assert result["source"] == "app"

def test_analyze_neutral_text():
    """Test sentiment analysis with more neutral text"""
    review = {
        "text": "The product arrived on time. It's what I ordered.",
        "review_id": "test789"
    }
    response = client.post("/api/v1/analyze", json=review)
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result
    assert -1 <= result["normalized_score"] <= 1

def test_analyze_with_html():
    """Test sentiment analysis with HTML in the text"""
    review = {
        "text": "<div>This product is <strong>fantastic</strong>!</div>",
        "source": "website"
    }
    response = client.post("/api/v1/analyze", json=review)
    assert response.status_code == 200
    result = response.json()
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.5

def test_batch_with_mixed_truncation():
    """Test batch processing with some reviews requiring truncation"""
    batch_request = {
        "reviews": [
            {
                "text": "Great product, very satisfied!",
                "review_id": "batch1",
                "truncate": True
            },
            {
                "text": "This product is great. " * 200,  # Long text
                "review_id": "batch2",
                "truncate": True  # Allow truncation
            },
            {
                "text": "Disappointed with the quality.",
                "review_id": "batch3"
            }
        ],
        "truncate": False  # Global setting is false, but batch2 overrides it
    }
    
    response = client.post("/api/v1/analyze/batch", json=batch_request)
    assert response.status_code == 200
    results = response.json()["results"]
    
    # Check we got 3 results
    assert len(results) == 3
    
    # First review should be positive and not truncated
    assert results[0]["sentiment"] == "positive"
    assert results[0]["truncated"] == False
    
    # Second review should be positive and truncated
    assert results[1]["sentiment"] == "positive"
    assert results[1]["truncated"] == True
    
    # Third review should be negative and not truncated
    assert results[2]["sentiment"] == "negative"
    assert results[2]["truncated"] == False

def test_empty_text():
    """Test error handling with empty text"""
    review = {
        "text": "",
        "review_id": "test789",
        "source": "social_media"
    }
    response = client.post("/api/v1/analyze", json=review)
    assert response.status_code == 422  # Validation error

def test_missing_text():
    """Test error handling with missing text field"""
    review = {
        "review_id": "test999",
        "source": "social_media"
    }
    response = client.post("/api/v1/analyze", json=review)
    assert response.status_code == 422  # Validation error

def test_invalid_batch():
    """Test error handling with empty batch"""
    batch_request = {
        "reviews": []
    }
    response = client.post("/api/v1/analyze/batch", json=batch_request)
    assert response.status_code == 422  # Validation error

def test_large_text_with_truncation():
    """Test with a very large text input that exceeds model's token limit, with truncation enabled"""
    # Generate a long text (approximately 1000 words)
    long_text = "This product is great. " * 200
    
    review = {
        "text": long_text,
        "review_id": "long1",
        "truncate": True  # Explicitly enable truncation
    }
    
    response = client.post("/api/v1/analyze", json=review)
    assert response.status_code == 200  # Should succeed with truncation
    result = response.json()
    
    # Verify we got valid sentiment results despite the long text
    assert result["sentiment"] == "positive"
    assert result["truncated"] == True  # Should indicate truncation happened
    assert result["confidence"] > 0.5

def test_large_text_without_truncation():
    """Test with a very large text input that exceeds model's token limit, with truncation disabled"""
    # Generate a long text (approximately 1000 words)
    long_text = "This product is great. " * 200
    
    review = {
        "text": long_text,
        "review_id": "long2",
        "truncate": False  # Disable truncation
    }
    
    # This should now raise a 400 error due to the text being too long
    response = client.post("/api/v1/analyze", json=review)
    assert response.status_code == 400  # Should fail with 400 Bad Request
    
    # Check the error message
    error_detail = response.json()["detail"]
    assert "too long" in error_detail or "exceeds" in error_detail