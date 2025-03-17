# Review Sentiment Analysis Service

A microservice for analyzing sentiment in customer reviews with a focus on ML/Ops.

## Project Overview

This project implements a sentiment analysis service that can be used as part of a larger system to analyze customer reviews. It was developed as a solution for the Junior Backend Engineer (ML/Ops Focus) technical assessment.

The complete solution consists of two parts:

  Part 1: Conceptual planning and architecture design (provided in the PDF file: Conceptual_planning.pdf) <br/>
  Part 2: A practical microservice implementation with API endpoints for sentiment analysis (contained in this repository)

### Features

- RESTful API for sentiment analysis
- Pre-trained sentiment analysis model (DistilBERT)
- Text preprocessing for cleaning and normalizing review text
- Batch processing capabilities
- Error handling for long texts
- API documentation via Swagger UI
- Docker support

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Conda (recommended) or pip (Python package manager)
- Docker (optional, for containerized deployment)

### Option 1: Local Setup with Conda (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/haibkhn/sentiment-analysis-service
   cd sentiment-analysis-service
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate sentiment-analysis
   ```

3. Download the sentiment analysis model:
   ```bash
   python scripts/download_model.py
   ```

4. Start the service:
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at http://localhost:8000

### Option 2: Local Setup with Pip

1. Clone the repository:
   ```bash
   git clone https://github.com/haibkhn/sentiment-analysis-service
   cd sentiment-analysis-service
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the sentiment analysis model:
   ```bash
   python scripts/download_model.py
   ```

5. Start the service:
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at http://localhost:8000

### Option 3: Docker Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/haibkhn/sentiment-analysis-service
   cd sentiment-analysis-service
   ```

2. Build the Docker image:
   ```bash
   docker build -t sentiment-analysis-service .
   ```

3. Run the container:
   ```bash
   docker run -p 8000:8000 sentiment-analysis-service
   ```

The API will be available at http://localhost:8000

## API Documentation

### Endpoints

#### POST /api/v1/analyze

Analyzes the sentiment of a single review text.

**Request Body:**

```json
{
  "text": "This product exceeded my expectations. I love it!",
  "review_id": "123456",
  "source": "website",
  "truncate": true
}
```

**Sample Response:**

```json
{
  "sentiment": "positive",
  "confidence": 0.989,
  "normalized_score": 0.978,
  "detailed_scores": {
    "POSITIVE": 0.989,
    "NEGATIVE": 0.011
  },
  "review_id": "123456",
  "source": "website",
  "truncated": false
}
```

#### POST /api/v1/analyze/batch

Analyzes multiple reviews in a single request.

**Request Body:**

```json
{
  "reviews": [
    {
      "text": "Great product, very satisfied!",
      "review_id": "batch1",
      "source": "website"
    },
    {
      "text": "Disappointed with the quality.",
      "review_id": "batch2",
      "source": "app"
    }
  ],
  "truncate": true
}
```

**Response:**

Provides an array of sentiment analysis results, with the same structure as the single review analysis.

#### GET /api/v1/health

Simple health check endpoint.

**Response:**

```json
{
  "status": "healthy"
}
```

## Testing the API

### Automated Testing

Run the automated tests using pytest:

```bash
# Run all tests. If you haven't downloaded the model yet, run `python scripts/download_model.py` first. Otherwise, wait a few minutes for the model to download.
python -m pytest

# Run specific test files
python -m pytest tests/test_api.py
python -m pytest tests/test_preprocessing.py 
python -m pytest tests/test_sentiment_model.py
```

### Manual Testing with Swagger UI

The easiest way to test the API is using the built-in Swagger UI:

1. Start the server (see "Setup Instructions" above)
2. Open http://localhost:8000/docs in your browser
3. Try out the different endpoints through the interactive UI

### Manual Testing with cURL

#### Health Check:
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

#### Sentiment Analysis (Positive):
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is amazing! I absolutely love it.",
    "review_id": "test123",
    "source": "website"
  }'
```

#### Sentiment Analysis (Negative):
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is terrible. I regret buying this product.",
    "review_id": "test456",
    "source": "app"
  }'
```

#### Batch Analysis:
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      {
        "text": "Great product, very satisfied!",
        "review_id": "batch1",
        "source": "website"
      },
      {
        "text": "Disappointed with the quality.",
        "review_id": "batch2",
        "source": "app"
      }
    ]
  }'
```

### Manual Testing with Python Requests

You can also test using Python's requests library:

```python
import requests
import json

# Test health check
response = requests.get("http://localhost:8000/api/v1/health")
print(response.json())

# Test sentiment analysis
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={
        "text": "This product is amazing!",
        "review_id": "test123",
        "source": "website"
    }
)
print(json.dumps(response.json(), indent=2))
```

## Project Structure

```
sentiment-analysis-service/
├── app/
│   ├── __init__.py
│   ├── main.py            # Main FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── sentiment.py   # Sentiment analysis model
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py   # API endpoints
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py # Text preprocessing
├── tests/
│   ├── __init__.py
│   ├── test_api.py        # API tests
│   ├── test_preprocessing.py # Preprocessing tests
│   └── test_sentiment_model.py # Model tests
├── scripts/
│   └── download_model.py  # Script to download and cache model
├── Dockerfile             # For containerization
├── environment.yml        # Conda environment specification
├── requirements.txt       # Dependencies
├── .gitignore             # Files to ignore in Git
└── README.md              # This file
```

## Design Decisions and Assumptions

- **Model Selection**: Using DistilBERT for a good balance of accuracy and performance
- **API Design**: RESTful API with JSON for simplicity and broad compatibility
- **Error Handling**: Dedicated handling for cases like token limits
- **Preprocessing**: Flexible preprocessing to handle various input formats
- **Docker Support**: Configuration for containerized deployment

## Future Improvements

- Multi-language support
- More fine-grained sentiment analysis (beyond binary classification)
- Topic extraction from reviews
- Caching frequent patterns for improved performance
- Advanced batch processing with background workers