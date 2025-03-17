from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from app.models.sentiment import SentimentAnalyzer
from app.utils.preprocessing import preprocess_text

# Initialize router
router = APIRouter()

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

# Define request/response models
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The review text to analyze")
    review_id: Optional[str] = Field(None, description="Optional review identifier")
    source: Optional[str] = Field(None, description="Source of the review (website, app, social media, etc.)")
    truncate: bool = Field(True, description="Whether to truncate text that exceeds model's token limit")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Overall sentiment (positive/negative)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    normalized_score: float = Field(..., description="Normalized score from -1 (very negative) to 1 (very positive)")
    detailed_scores: Dict[str, float] = Field(..., description="Detailed sentiment scores")
    review_id: Optional[str] = None
    source: Optional[str] = None
    truncated: bool = Field(False, description="Whether the input text was truncated before analysis")

class BatchReviewRequest(BaseModel):
    reviews: List[ReviewRequest] = Field(..., min_length=1, max_length=100, 
                                      description="List of reviews to analyze")
    truncate: bool = Field(True, description="Global setting for whether to truncate texts (can be overridden in individual reviews)")

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

@router.post("/analyze", response_model=SentimentResponse, 
             summary="Analyze review sentiment")
async def analyze_sentiment(review: ReviewRequest = Body(...)):
    """
    Analyze the sentiment of a customer review.
    
    - **text**: The review text to analyze
    - **review_id**: Optional identifier for the review
    - **source**: Optional source of the review
    - **truncate**: Whether to truncate text that exceeds model's token limit (default: true)
    
    Returns sentiment analysis results including:
    - Overall sentiment (positive/negative)
    - Confidence score
    - Normalized score from -1 (very negative) to 1 (very positive)
    - Detailed sentiment scores
    - Whether the text was truncated
    
    If truncate=false and text exceeds the model's token limit, returns 400 Bad Request.
    """
    try:
        # Check if text might exceed token limit (rough estimation)
        # 250 words is a conservative estimation to stay under 512 tokens
        words = review.text.split()
        was_truncated = len(words) > 250
        
        # If text is too long and truncation is disabled, return error
        if was_truncated and not review.truncate:
            raise HTTPException(
                status_code=400, 
                detail="Text exceeds model's token limit (>250 words / ~512 tokens) and truncation is disabled"
            )
        
        # Preprocess the text
        processed_text = preprocess_text(review.text)
        
        # Truncate if necessary and enabled
        if was_truncated and review.truncate:
            processed_text = " ".join(processed_text.split()[:500])
        
        # Analyze sentiment
        analysis_results = sentiment_analyzer.analyze(processed_text)
        
        # Create response
        response = SentimentResponse(
            **analysis_results,
            review_id=review.review_id,
            source=review.source,
            truncated=was_truncated and review.truncate
        )
        
        return response
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@router.post("/analyze/batch", response_model=BatchSentimentResponse, 
            summary="Analyze multiple reviews in batch")
async def analyze_batch(request: BatchReviewRequest = Body(...)):
    """
    Analyze the sentiment of multiple customer reviews in batch.
    
    - **reviews**: List of reviews to analyze (max 100)
    - **truncate**: Global setting for whether to truncate texts (default: true)
    
    Each review can override the global truncate setting with its own truncate field.
    
    Returns a list of sentiment analysis results for each review.
    If any review exceeds the token limit and has truncation disabled, a 400 error is returned.
    """
    try:
        results = []
        
        for review in request.reviews:
            # Use global truncate setting if not specified in individual review
            if not hasattr(review, 'truncate'):
                review.truncate = request.truncate
                
            # Check if text might exceed token limit (rough estimation)
            # 250 words is a conservative estimation to stay under 512 tokens
            words = review.text.split()
            was_truncated = len(words) > 250
            
            # If text is too long and truncation is disabled, return error
            if was_truncated and not review.truncate:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Review {review.review_id} exceeds model's token limit (>250 words / ~512 tokens) and truncation is disabled"
                )
            
            # Preprocess the text first
            processed_text = preprocess_text(review.text)
            
            # Truncate if necessary and enabled
            if was_truncated and review.truncate:
                # More conservative truncation - aim for ~250 words which should be under 512 tokens
                processed_text = " ".join(processed_text.split()[:250])
                
            # Check if still too long after preprocessing (safety check)
            if len(processed_text.split()) > 250 and not review.truncate:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Text is too long after preprocessing and truncation is disabled"
                )
            
            # Analyze sentiment
            analysis_results = sentiment_analyzer.analyze(processed_text)
            
            # Create response item
            results.append(SentimentResponse(
                **analysis_results,
                review_id=review.review_id,
                source=review.source,
                truncated=was_truncated and review.truncate
            ))
        
        return BatchSentimentResponse(results=results)
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing batch sentiment: {str(e)}")

@router.get("/health", summary="Check service health")
async def health_check():
    """
    Simple health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}