"""
SCRIPT 2/5: augmenter.py - Context Enhancement and Prompt Engineering for Shoe RAG Pipeline

Colab - https://colab.research.google.com/drive/1rq-ywjykHBw7xPXCmd3DmZdK6T9bhDtA?usp=sharing

This script handles the AUGMENTATION phase of the RAG pipeline, including:
- Query classification and analysis
- Context formatting and enhancement
- Prompt engineering with different strategies
- Advanced prompt templates for different query types

Key Concepts:
- Prompt Engineering: Designing effective prompts to guide LLM responses
- Context Enhancement: Structuring retrieved data for optimal LLM understanding
- Query Classification: Determining intent to apply appropriate prompt strategies
- Template-based Prompting: Using structured templates for consistent results

Required Dependencies:
- typing: Type hints for better code structure
- enum: For defining query types

Commands to run:
# Test query classification
python augmenter.py --query "recommend running shoes for men" --classify-only --search-type auto

# Test context formatting with real data
python augmenter.py --query "show me casual sneakers" --test-formatting

# Generate prompt for recommendation query with real data
python augmenter.py --query "recommend comfortable shoes" --generate-prompt

# Generate prompt for search query with real data
python augmenter.py --query "blue sneakers" --generate-prompt

# Test with image search context
python augmenter.py --query "hf_shoe_images/shoe_0000.jpg" --search-type image --generate-prompt

# Test with auto search type detection
python augmenter.py --query "recommend shoes" --generate-prompt

# Use custom database settings
python augmenter.py --query "sneakers" --generate-prompt --database "myntra_shoes_db" --table-name "myntra_shoes_table"
"""

import argparse
from enum import Enum
from typing import Any, Dict, List

# Import retriever components
from retriever import MyntraShoesEnhanced, run_shoes_search


class QueryType(Enum):
    """Query types for different shoe-related interactions."""

    RECOMMENDATION = "recommendation"
    SEARCH = "search"


class SimpleShoePrompts:
    """AUGMENTATION: Simplified prompt system for shoe RAG with context enhancement."""

    def __init__(self):
        self.system_prompts = {
            "recommendation": """You are a helpful assistant. Choose from the given shoe options and give a short, simple recommendation. Do not make up any information.""",
            "search": """You are a knowledgeable shoe assistant. Help customers understand the available shoe options 
that match their search criteria, providing detailed information about features and benefits.""",
        }

    def classify_query(self, query: str) -> QueryType:
        """Classify query into recommendation or search type."""
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["recommend", "suggest", "best", "need", "looking for"]
        ):
            return QueryType.RECOMMENDATION
        else:
            return QueryType.SEARCH

    def format_shoes_context(self, shoes: List[Dict[str, Any]]) -> str:
        """AUGMENTATION: Format retrieved shoes into readable context for LLM."""
        formatted_shoes = []
        for i, shoe in enumerate(shoes, 1):
            # Keep it simple - just basic info
            product_type = shoe.get("product_type", "Shoe")
            gender = shoe.get("gender", "")

            if gender:
                shoe_name = f"{product_type} for {gender}"
            else:
                shoe_name = product_type

            # Add basic color info if available
            color = shoe.get("color", "")
            if color and color not in ["None", None, ""]:
                shoe_name += f" ({color})"

            formatted_shoes.append(f"{i}. {shoe_name}")

        return "\n".join(formatted_shoes)

    def generate_prompt(
        self, query: str, shoes: List[Dict[str, Any]], search_type: str = "text"
    ) -> str:
        """AUGMENTATION: Generate complete prompt based on query type and retrieved context."""
        # If it's an image search, always treat as search query type
        if search_type == "image":
            query_type = QueryType.SEARCH
        else:
            query_type = self.classify_query(query)

        system_prompt = self.system_prompts[query_type.value]
        context = self.format_shoes_context(shoes)

        if query_type == QueryType.RECOMMENDATION:
            # Add a summary to guide recommendations
            intent_summary = (
                f"Based on the query, the user is likely looking for {query.lower()}."
            )

            user_prompt = f"""{intent_summary}

Available Options:
{context}

Your task:
- Recommend the best option(s) that align most closely with the query.
- Reference specific attributes (e.g., gender, product type, color, or other features) in your reasoning.
- Avoid adding details not provided in the context.

Provide your recommendation in 2-3 sentences."""

        else:  # SEARCH
            user_prompt = f"""Here are shoes matching: "{query}"

Search Results:
{context}

Explain how well these shoes meet the search criteria and highlight their relevant features."""

        return f"{system_prompt}\n\n{user_prompt}"


def detect_search_type(search_query) -> str:
    """Auto-detect search type based on query content (matches retriever.py logic)."""
    # Auto-detect search type
    if isinstance(search_query, str):
        if search_query.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            # Image file path
            return "image"
        else:
            # Text query
            return "text"
    elif hasattr(search_query, "save"):  # PIL Image object
        return "image"
    else:
        return "text"


def get_real_shoes_data(
    query: str,
    search_type: str = "text",
    database: str = "myntra_shoes_db",
    table_name: str = "myntra_shoes_table",
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Get real shoes data from retriever for testing purposes."""

    try:
        results, _ = run_shoes_search(
            database=database,
            table_name=table_name,
            schema=MyntraShoesEnhanced,
            search_query=query,
            limit=limit,
            search_type=search_type,
            output_folder="output_augmenter",
        )
        return results
    except Exception as e:
        raise Exception(
            f"Could not retrieve real data: {e}. Please ensure the database is set up correctly."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augmentation component for Shoe RAG Pipeline"
    )
    parser.add_argument("--query", type=str, required=True, help="Query to process")
    parser.add_argument(
        "--search-type",
        choices=["auto", "text", "image"],
        default="auto",
        help="Search type for prompt generation (auto-detect or force specific type)",
    )
    parser.add_argument(
        "--classify-only", action="store_true", help="Only classify the query type"
    )
    parser.add_argument(
        "--test-formatting",
        action="store_true",
        help="Test context formatting with real data",
    )
    parser.add_argument(
        "--generate-prompt", action="store_true", help="Generate complete prompt"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="myntra_shoes_db",
        help="Database path for real data",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="myntra_shoes_table",
        help="Table name for real data",
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="Number of shoes to retrieve for testing"
    )

    args = parser.parse_args()

    # Initialize prompt manager
    prompt_manager = SimpleShoePrompts()

    # Auto-detect search type if needed
    if args.search_type == "auto":
        detected_search_type = detect_search_type(args.query)
        if detected_search_type == "image":
            print(f"🖼️  Detected image search: {args.query}")
        else:
            print(f"📝 Detected text search: {args.query}")
    else:
        detected_search_type = args.search_type

    # Classify query
    query_type = prompt_manager.classify_query(args.query)
    print("=" * 60)
    print("📝 AUGMENTATION RESULTS")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Query Type: {query_type.value}")
    print(f"Search Type: {args.search_type} → {detected_search_type}")

    if args.classify_only:
        print("\n🎯 Query Classification Complete!")

    elif args.test_formatting:
        # Test context formatting with real data
        shoes_data = get_real_shoes_data(
            query=args.query,
            search_type=detected_search_type,
            database=args.database,
            table_name=args.table_name,
            limit=args.limit,
        )

        formatted_context = prompt_manager.format_shoes_context(shoes_data)

        print(f"\n📊 Context Formatting Test (Real Data):")
        print("-" * 40)
        print(f"Real Shoes Data:")
        for i, shoe in enumerate(shoes_data, 1):
            print(f"  {i} {shoe}")

        print("\nFormatted Context:")
        print("-" * 40)
        print(formatted_context)

    elif args.generate_prompt:
        # Generate complete prompt with real data
        shoes_data = get_real_shoes_data(
            query=args.query,
            search_type=detected_search_type,
            database=args.database,
            table_name=args.table_name,
            limit=args.limit,
        )

        complete_prompt = prompt_manager.generate_prompt(
            args.query, shoes_data, detected_search_type
        )

        print(f"\n🔍 Complete Prompt Generation (Real Data):")
        print("-" * 40)
        print("System Prompt:")
        print(prompt_manager.system_prompts[query_type.value])

        print("\nFormatted Context:")
        formatted_context = prompt_manager.format_shoes_context(shoes_data)
        print(formatted_context)

        print("\nComplete Prompt:")
        print("-" * 40)
        print(complete_prompt)

    else:
        # Show all information with real data
        shoes_data = get_real_shoes_data(
            query=args.query,
            search_type=detected_search_type,
            database=args.database,
            table_name=args.table_name,
            limit=args.limit,
        )

        formatted_context = prompt_manager.format_shoes_context(shoes_data)
        complete_prompt = prompt_manager.generate_prompt(
            args.query, shoes_data, detected_search_type
        )

        print(f"\n📊 Context Formatting (Real Data):")
        print("-" * 40)
        print(formatted_context)

        print("\n🔍 Complete Prompt:")
        print("-" * 40)
        print(complete_prompt)

        print(f"\n✅ Augmentation Complete! (Used Real Data)")