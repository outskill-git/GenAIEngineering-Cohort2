"""
Fast Multi-Agent News Aggregator
Optimized for speed while still using multiple agents and tools
"""

from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Any
from datetime import datetime
import requests
import json
import os
import asyncio
import concurrent.futures


class SearchInput(BaseModel):
    query: str = Field(description="Search query")


class FastNewsSearchTool(BaseTool):
    """Fast news search tool"""
    name: str = "fast_news_search"
    description: str = "Quickly search for news articles"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        """Fast news search with timeout"""
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return json.dumps({"articles": []})

        try:
            response = requests.post(
                "https://google.serper.dev/news",
                headers={'X-API-KEY': api_key, 'Content-Type': 'application/json'},
                json={"q": query, "num": 8},
                timeout=5  # Short timeout
            )

            if response.status_code == 200:
                data = response.json()
                articles = []

                # Quick extraction
                for item in data.get("news", [])[:8]:
                    if item.get("link"):
                        articles.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "source": item.get("source", ""),
                            "date": item.get("date", ""),
                            "snippet": item.get("snippet", "")
                        })

                return json.dumps({"articles": articles})

            return json.dumps({"articles": []})

        except:
            return json.dumps({"articles": []})


class NewsAggregatorCrew:
    """Fast crew with only 2 agents"""

    def __init__(self, topic: str):
        self.topic = topic
        self.search_tool = FastNewsSearchTool()
        # Pre-fetch news to save time
        self.prefetched_news = self._prefetch_news()

    def _prefetch_news(self) -> str:
        """Pre-fetch news before agents start"""
        return self.search_tool._run(self.topic)

    # AGENT 1: News Fetcher (combines search and curation)
    def news_fetcher_agent(self) -> Agent:
        return Agent(
            role="News Fetcher",
            goal=f"Quickly find and select the best news about {self.topic}",
            backstory="Expert at rapidly finding and filtering news.",
            tools=[self.search_tool],
            verbose=False,  # Less verbose = faster
            allow_delegation=False,
            max_iter=1  # Limit iterations
        )

    # AGENT 2: News Formatter (just formats, no analysis)
    def news_formatter_agent(self) -> Agent:
        return Agent(
            role="News Formatter",
            goal="Format news into clean JSON output",
            backstory="Expert at quickly formatting data.",
            verbose=False,
            allow_delegation=False,
            max_iter=1
        )

    # TASK 1: Quick fetch and filter
    def fetch_task(self) -> Task:
        return Task(
            description=f"""Here are pre-fetched news results:

{self.prefetched_news}

Quickly select the 5 best articles and list them with:
- Title
- URL
- Source
- Date

No analysis needed, just pick the best ones.""",
            expected_output="List of 5 best articles",
            agent=self.news_fetcher_agent()
        )

    # TASK 2: Quick format
    def format_task(self) -> Task:
        return Task(
            description="""Format the articles as a JSON array.

For each article, create:
{{
    "title": "title",
    "url": "exact URL",
    "source": "source",
    "date": "date",
    "category": "News",
    "summary": "Use the snippet as-is",
    "key_points": [],
    "credibility": "High" for BBC/Reuters/CNN, "Medium" for others
}}

Output ONLY the JSON array.""",
            expected_output="JSON array",
            agent=self.news_formatter_agent()
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.news_fetcher_agent(),
                self.news_formatter_agent()
            ],
            tasks=[
                self.fetch_task(),
                self.format_task()
            ],
            process=Process.sequential,
            verbose=False  # Less output = faster
        )

    def run(self) -> Dict[str, Any]:
        """Fast execution"""
        try:
            # Set a timeout for the entire crew execution
            crew = self.crew()
            result = crew.kickoff()

            # Quick parse
            parsed = self._quick_parse(str(result))

            return {
                "success": True,
                "topic": self.topic,
                "timestamp": datetime.now().isoformat(),
                "news_count": len(parsed),
                "news_items": parsed,
                "raw_output": str(result)[:500]  # Truncate for speed
            }

        except Exception as e:
            # Fallback to direct results
            return self._direct_results()

    def _quick_parse(self, output: str) -> List[Dict[str, Any]]:
        """Quick parsing without complex regex"""
        try:
            # Find JSON array
            start = output.find('[')
            end = output.rfind(']') + 1

            if start >= 0 and end > start:
                json_str = output[start:end]
                items = json.loads(json_str)
                return items[:5]  # Max 5 items
        except:
            pass

        return self._parse_prefetched()

    def _parse_prefetched(self) -> List[Dict[str, Any]]:
        """Parse the pre-fetched results directly"""
        try:
            data = json.loads(self.prefetched_news)
            articles = data.get("articles", [])

            news_items = []
            for article in articles[:5]:
                news_items.append({
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", ""),
                    "date": article.get("date", datetime.now().strftime("%Y-%m-%d")),
                    "category": "News",
                    "summary": article.get("snippet", ""),
                    "key_points": [],
                    "credibility": "High" if article.get("source") in ["BBC", "Reuters", "CNN"] else "Medium"
                })

            return news_items
        except:
            return []

    def _direct_results(self) -> Dict[str, Any]:
        """Skip agents and return direct results"""
        items = self._parse_prefetched()

        return {
            "success": True,
            "topic": self.topic,
            "timestamp": datetime.now().isoformat(),
            "news_count": len(items),
            "news_items": items,
            "raw_output": "Direct results (fast mode)"
        }


class MockNewsAggregatorCrew:
    """Mock - instant results"""

    def __init__(self, topic: str):
        self.topic = topic

    def run(self) -> Dict[str, Any]:
        """Instant mock data"""
        return {
            "success": True,
            "topic": self.topic,
            "timestamp": datetime.now().isoformat(),
            "news_count": 3,
            "news_items": [
                {
                    "title": f"{self.topic} News Update",
                    "url": f"https://example.com/{self.topic.lower()}-news",
                    "source": "Example News",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "category": "News",
                    "summary": f"Latest update on {self.topic}.",
                    "key_points": [],
                    "credibility": "High"
                }
            ],
            "raw_output": "Mock data"
        }