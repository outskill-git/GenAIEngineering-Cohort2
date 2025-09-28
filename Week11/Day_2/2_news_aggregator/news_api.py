"""
Simplified FastAPI Backend for News Aggregator
Expects JSON output from CrewAI agents
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import logging
import dotenv
import json

# Import the crew module
from news_crew import NewsAggregatorCrew, MockNewsAggregatorCrew

dotenv.load_dotenv()

# Set up environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY", "")
os.environ["OPENAI_API_BASE"] = 'https://openrouter.ai/api/v1'
os.environ["OPENAI_BASE_URL"] = 'https://openrouter.ai/api/v1'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="News Aggregator API",
    description="AI-powered news fetching using CrewAI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class NewsItem(BaseModel):
    """Model for a single news item"""
    title: str
    source: str
    date: str
    url: Optional[str] = ""
    category: str = "General"
    summary: str
    key_points: List[str] = []
    credibility: str = "Medium"


class NewsResponse(BaseModel):
    """Response model for news search"""
    success: bool
    topic: str
    news_count: int
    news_items: List[NewsItem]
    message: Optional[str] = None


# API Endpoints
@app.get("/", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "News Aggregator API",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/news/{topic}", response_model=NewsResponse, tags=["news"])
async def get_news(
    topic: str,
    mock: bool = Query(False, description="Use mock data for testing")
):
    """
    Fetch news articles for a specific topic
    """
    try:
        logger.info(f"Fetching news for topic: {topic} (mock={mock})")
        print(f"Fetching news for topic: {topic} (mock={mock})")

        # Create appropriate crew instance
        if mock:
            crew = MockNewsAggregatorCrew(topic)
        else:
            crew = NewsAggregatorCrew(topic)

        # Run the crew and get results
        result = crew.run()

        # The crew now returns a dictionary with all needed data
        if result.get("success") and result.get("news_items"):
            # Convert to NewsItem objects
            news_items = []
            for item in result["news_items"]:
                news_items.append(NewsItem(
                    title=item.get("title", "Untitled"),
                    source=item.get("source", "Unknown"),
                    date=item.get("date", datetime.now().strftime("%Y-%m-%d")),
                    url=item.get("url", ""),
                    category=item.get("category", "General"),
                    summary=item.get("summary", "No summary available"),
                    key_points=item.get("key_points", []),
                    credibility=item.get("credibility", "Medium")
                ))

            return NewsResponse(
                success=True,
                topic=topic,
                news_count=len(news_items),
                news_items=news_items,
                message=f"Successfully fetched {len(news_items)} news items"
            )
        else:
            # Log the issue
            logger.warning(f"No news items returned for topic: {topic}")
            logger.warning(f"Result: {result}")

            return NewsResponse(
                success=False,
                topic=topic,
                news_count=0,
                news_items=[],
                message="No news items found - crew may have failed to generate content"
            )

    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        return NewsResponse(
            success=False,
            topic=topic,
            news_count=0,
            news_items=[],
            message=f"Error: {str(e)}"
        )



@app.get("/api/topics/trending", tags=["topics"])
async def get_trending_topics():
    """Get list of trending topics for quick selection"""
    return {
        "topics": [
            "Artificial Intelligence",
            "Climate Change",
            "Stock Market",
            "Technology",
            "Healthcare",
            "Space Exploration",
            "Renewable Energy",
            "Cybersecurity",
            "Economic Policy",
            "Scientific Research"
        ]
    }


@app.get("/api/topics/recent", tags=["topics"])
async def get_recent_searches(limit: int = Query(10, ge=1, le=50)):
    """Get recently searched topics (placeholder)"""
    return {
        "recent_topics": [
            {"topic": "AI Ethics", "timestamp": "2025-01-15T10:30:00"},
            {"topic": "Climate Summit", "timestamp": "2025-01-15T09:45:00"},
            {"topic": "Tech Stocks", "timestamp": "2025-01-15T08:20:00"}
        ][:limit]
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Not found", "message": "The requested resource was not found"}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "An unexpected error occurred"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting News Aggregator API...")
    logger.info("Expecting JSON output from CrewAI agents")

    uvicorn.run(
        "news_api:app",
        host="0.0.0.0",
        port=9321,
        reload=True,
        log_level="info"
    )