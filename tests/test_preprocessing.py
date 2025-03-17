import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.preprocessing import preprocess_text, clean_review_batch

def test_basic_functionality():
    """Test that the preprocessing function handles basic cases"""
    # Test lowercase conversion
    assert preprocess_text("HELLO WORLD") == "hello world"
    
    # Test whitespace normalization
    assert preprocess_text("too    many    spaces") == "too many spaces"
    
    # Test HTML removal
    assert preprocess_text("<p>text with html</p>") == "text with html"
    
    # Test URL removal
    assert preprocess_text("check out https://example.com") == "check out"
    
    # Test special character removal
    assert preprocess_text("text with #hashtag and @mention") == "text with hashtag and mention"
    
    # Test punctuation preservation
    assert preprocess_text("This has, some. punctuation!") == "this has, some. punctuation!"

def test_batch_processing():
    """Test batch processing of multiple reviews"""
    reviews = [
        "This is GREAT!",
        "<p>HTML content</p>",
        "URL: https://example.com"
    ]
    
    processed = clean_review_batch(reviews)
    
    assert len(processed) == 3
    assert processed[0] == "this is great!"
    assert processed[1] == "html content"
    assert processed[2] == "url"

def test_real_world_examples():
    """Test with real-world review examples"""
    examples = [
        # Regular review
        "I bought this product last week and I'm very impressed with the quality. Would definitely recommend!",
        # Review with HTML
        "<div class='review-content'>This product is <strong>fantastic</strong>! I would buy it again.</div>",
        # Review with URL
        "Check out my full review with photos here: https://reviewsite.com/review123. Overall, very satisfied!",
        # Social media style review
        "Just got my new #ProductX and it's absolutely AMAZING! 😍 @CompanyName has outdone themselves! #HappyCustomer",
        # Product rating review
        "Product Rating: ★★★★☆ (4/5)\n- Pros: Fast delivery, good quality\n- Cons: Slightly expensive"
    ]
    
    # We're not checking exact output strings, just verifying basic functionality
    processed = clean_review_batch(examples)
    
    for p in processed:
        # Verify text is lowercase
        assert p == p.lower()
        
        # Verify HTML tags are removed
        assert "<" not in p
        assert ">" not in p
        
        # Verify URLs are removed
        assert "http" not in p
        assert "www." not in p
        
        # Verify special characters are removed
        assert "#" not in p
        assert "@" not in p
        assert "★" not in p
        
        # Verify text isn't empty after processing
        assert len(p) > 0

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])