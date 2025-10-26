"""
SCRIPT 1/5: retriever.py - Vector Search and Data Management for Shoe RAG Pipeline

Colab - https://colab.research.google.com/drive/1rq-ywjykHBw7xPXCmd3DmZdK6T9bhDtA?usp=sharing

This script handles the RETRIEVAL phase of the RAG pipeline, including:
- Setting up vector embeddings using OpenAI CLIP model
- Creating and managing LanceDB vector database
- Loading data from HuggingFace datasets
- Parsing shoe attributes from text descriptions
- Performing vector similarity search on shoes

Key Concepts:
- Vector Embeddings: Converting images/text to numerical vectors for similarity search
- LanceDB: Vector database for storing and querying embeddings
- Pydantic Models: Data validation and serialization
- Similarity Search: Finding similar items based on vector distance

Required Dependencies:
- lancedb: Vector database
- datasets: HuggingFace datasets
- PIL: Image processing
- pandas: Data manipulation

Commands to run:
# Setup database from HuggingFace dataset
python retriever.py --setup-db --database myntra_shoes_db --table myntra_shoes_table --sample-size 500

# Test vector search with text query
python retriever.py --query "running shoes for men" --limit 5

# Test vector search with image
python retriever.py --query "hf_shoe_images/shoe_0000.jpg" --search-type image
"""

import argparse
import os
import re
from pathlib import Path
from random import sample
from typing import Any, Dict, List, Optional

import lancedb
import pandas as pd
from datasets import load_dataset
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector
from PIL import Image


def register_model(model_name: str) -> Any:
    """Register a model with the given name using LanceDB's EmbeddingFunctionRegistry."""
    registry = EmbeddingFunctionRegistry.get_instance()
    model = registry.get(model_name).create()
    return model


# Register the OpenAI CLIP model for vector embeddings
clip = register_model("open-clip")


class MyntraShoesEnhanced(LanceModel):
    """Enhanced Myntra Shoes Schema with product metadata for vector storage."""

    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()

    # Core product information extracted from text
    product_id: Optional[str] = None
    description: Optional[str] = None
    product_type: Optional[str] = None
    gender: Optional[str] = None
    color: Optional[str] = None

    # Shoe-specific attributes
    toe_shape: Optional[str] = None
    pattern: Optional[str] = None
    fastening: Optional[str] = None
    shoe_width: Optional[str] = None
    ankle_height: Optional[str] = None
    insole: Optional[str] = None
    sole_material: Optional[str] = None

    @property
    def image(self):
        if isinstance(self.image_uri, str) and os.path.exists(self.image_uri):
            return Image.open(self.image_uri)
        elif hasattr(self.image_uri, "save"):  # PIL Image object
            return self.image_uri
        else:
            # Return a placeholder or handle the case appropriately
            return None


def parse_shoe_attributes(text: str) -> dict:
    """Parse shoe attributes from the text description for structured storage."""
    attributes = {}

    # Extract product type (Men/Women + product type)
    if text.startswith("Men "):
        attributes["gender"] = "Men"
        attributes["product_type"] = text.split("Men ")[1].split(".")[0].strip()
    elif text.startswith("Women "):
        attributes["gender"] = "Women"
        attributes["product_type"] = text.split("Women ")[1].split(".")[0].strip()
    else:
        attributes["gender"] = None
        attributes["product_type"] = text.split(".")[0].strip()

    # Extract structured attributes using regex
    patterns = {
        "toe_shape": r"Toe Shape: ([^,]+)",
        "pattern": r"Pattern: ([^,]+)",
        "fastening": r"Fastening: ([^,]+)",
        "shoe_width": r"Shoe Width: ([^,]+)",
        "ankle_height": r"Ankle Height: ([^,]+)",
        "insole": r"Insole: ([^,]+)",
        "sole_material": r"Sole Material: ([^,\.]+)",
    }

    for attr, pattern in patterns.items():
        match = re.search(pattern, text)
        attributes[attr] = match.group(1).strip() if match else None

    # Extract color information (basic color detection)
    color_keywords = [
        "white",
        "black",
        "brown",
        "blue",
        "red",
        "green",
        "grey",
        "gray",
        "navy",
        "tan",
        "beige",
        "pink",
        "purple",
        "yellow",
        "orange",
    ]

    text_lower = text.lower()
    detected_colors = [color for color in color_keywords if color in text_lower]
    attributes["color"] = ", ".join(detected_colors) if detected_colors else None

    return attributes


def create_shoes_table_from_hf(
    database: str,
    table_name: str,
    dataset_name: str = "Harshgarg12/myntra_shoes_dataset",
    schema: Any = MyntraShoesEnhanced,
    mode: str = "overwrite",
    sample_size: int = 500,
    save_images: bool = True,
    images_dir: str = "hf_shoe_images",
) -> None:
    """Create vector database table with shoe data from Hugging Face dataset."""

    db = lancedb.connect(database)

    if table_name in db and mode != "overwrite":
        print(f"Table {table_name} already exists")
        return

    # Load dataset from Hugging Face
    print("Loading dataset from Hugging Face...")
    ds = load_dataset(dataset_name)
    train_data = ds["train"]

    # Sample data if needed
    if len(train_data) > sample_size:
        indices = sample(range(len(train_data)), sample_size)
        train_data = train_data.select(indices)

    print(f"Processing {len(train_data)} samples...")

    # Create images directory if saving images
    if save_images:
        os.makedirs(images_dir, exist_ok=True)

    # Prepare data for table creation
    table_data = []
    for i, item in enumerate(train_data):
        image = item["image"]
        text = item["text"]

        # Parse attributes from text
        attributes = parse_shoe_attributes(text)

        # Handle image
        if save_images:
            image_path = os.path.join(images_dir, f"shoe_{i:04d}.jpg")
            image.save(image_path, "JPEG")
            image_uri = image_path
        else:
            # Store PIL image directly (may cause issues with serialization)
            image_uri = image

        table_data.append(
            {
                "image_uri": image_uri,
                "product_id": f"hf_shoe_{i:04d}",
                "description": text,
                "product_type": attributes.get("product_type"),
                "gender": attributes.get("gender"),
                "color": attributes.get("color"),
                "toe_shape": attributes.get("toe_shape"),
                "pattern": attributes.get("pattern"),
                "fastening": attributes.get("fastening"),
                "shoe_width": attributes.get("shoe_width"),
                "ankle_height": attributes.get("ankle_height"),
                "insole": attributes.get("insole"),
                "sole_material": attributes.get("sole_material"),
            }
        )

    if table_data:
        if table_name in db:
            db.drop_table(table_name)

        table = db.create_table(table_name, schema=schema, mode="create")
        table.add(pd.DataFrame(table_data))
        print(f"Added {len(table_data)} shoes to table")
    else:
        print("No data to add")


def run_shoes_search(
    database: str,
    table_name: str,
    schema: Any,
    search_query: Any,
    limit: int = 6,
    output_folder: str = "output_retriever",
    search_type: str = "auto",  # "auto", "text", "image"
) -> tuple[list, str]:
    """RETRIEVAL: Run vector search on shoes and return detailed results."""

    # Clean output folder
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    else:
        os.makedirs(output_folder)

    db = lancedb.connect(database)
    table = db.open_table(table_name)

    # Determine search type and process query
    actual_search_type = search_type
    processed_query = search_query

    if search_type == "auto":
        # Auto-detect search type
        if isinstance(search_query, str):
            if search_query.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                # Image file path
                try:
                    processed_query = Image.open(search_query)
                    actual_search_type = "image"
                    print(f"🖼️  Detected image search: {search_query}")
                except Exception as e:
                    print(f"❌ Error loading image: {e}")
                    return [], "error"
            else:
                # Text query
                actual_search_type = "text"
                print(f"📝 Detected text search: {search_query}")
        elif hasattr(search_query, "save"):  # PIL Image object
            actual_search_type = "image"
            processed_query = search_query
            print("🖼️  Detected image search: PIL Image object")
        else:
            actual_search_type = "text"
            print(f"📝 Detected text search: {search_query}")

    elif search_type == "image":
        if isinstance(search_query, str):
            try:
                processed_query = Image.open(search_query)
                print(f"🖼️  Image search: {search_query}")
            except Exception as e:
                print(f"❌ Error loading image: {e}")
                return [], "error"
        elif hasattr(search_query, "save"):
            processed_query = search_query
            print("🖼️  Image search: PIL Image object")
        else:
            print("❌ Invalid image input for image search")
            return [], "error"

    else:  # text search
        actual_search_type = "text"
        print(f"📝 Text search: {search_query}")

    # Perform vector search
    try:
        results = table.search(processed_query).limit(limit).to_pydantic(schema)
    except Exception as e:
        print(f"❌ Search error: {e}")
        return [], "error"

    # Save images and collect metadata
    search_results = []
    for i, result in enumerate(results):
        image_path = os.path.join(output_folder, f"result_{i}.jpg")

        # Handle different image storage methods
        if result.image:
            result.image.save(image_path, "JPEG")
        else:
            print(f"Warning: No image available for result {i}")
            continue

        search_results.append(
            {
                "rank": i + 1,
                "product_id": result.product_id,
                "description": result.description,  # Remove truncation
                "product_type": result.product_type,
                "gender": result.gender,
                "color": result.color,
                "toe_shape": result.toe_shape,
                "pattern": result.pattern,
                "fastening": result.fastening,
                "image_path": image_path,
            }
        )

    return search_results, actual_search_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieval component for Shoe RAG Pipeline"
    )
    parser.add_argument(
        "--setup-db",
        action="store_true",
        help="Setup database from HuggingFace dataset",
    )
    parser.add_argument(
        "--query", type=str, help="Search query (text) or image file path"
    )
    parser.add_argument(
        "--search-type",
        choices=["auto", "text", "image"],
        default="auto",
        help="Force search type (default: auto-detect)",
    )
    parser.add_argument(
        "--limit", type=int, default=6, help="Number of results to return"
    )
    parser.add_argument(
        "--database", type=str, default="myntra_shoes_db", help="Database path"
    )
    parser.add_argument(
        "--table-name", type=str, default="myntra_shoes_table", help="Table name"
    )
    parser.add_argument(
        "--sample-size", type=int, default=500, help="Sample size for dataset"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="output_retriever",
        help="Output folder for results",
    )

    args = parser.parse_args()

    # Setup database if requested
    if args.setup_db:
        print("🔄 Setting up database from HuggingFace dataset...")
        create_shoes_table_from_hf(
            database=args.database,
            table_name=args.table_name,
            sample_size=args.sample_size,
            save_images=True,
        )
        print("✅ Database setup complete!")

    # Run search if query provided
    elif args.query:
        print("🔍 Running vector search...")
        results, search_type = run_shoes_search(
            database=args.database,
            table_name=args.table_name,
            schema=MyntraShoesEnhanced,
            search_query=args.query,
            limit=args.limit,
            output_folder=args.output_folder,
            search_type=args.search_type,
        )

        print("=" * 60)
        print("📊 RETRIEVAL RESULTS")
        print("=" * 60)
        print(f"Query: {args.query}")
        print(f"Search Type: {search_type}")
        print(f"Results Found: {len(results)}")
        print("\n👟 Retrieved Shoes:")
        for result in results:
            print(
                f"- {result['product_type']} ({result['gender']}) - {result['color']} - {result['pattern']}"
            )
            print(f"  📁 Image saved: {result['image_path']}")

        print(f"\n🖼️  Search results images saved in: {args.output_folder}/")

    else:
        print("❌ Please provide --setup-db to setup database or --query to search")
        print("\nExample usage:")
        print("  # Setup database")
        print("  python retriever.py --setup-db")
        print("  # Search with text")
        print("  python retriever.py --query 'running shoes for men'")
        print("  # Search with image")
        print("  python retriever.py --query 'path/to/image.jpg' --search-type image")