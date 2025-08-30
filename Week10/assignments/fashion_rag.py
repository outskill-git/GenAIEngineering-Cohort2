"""
Fashion RAG Pipeline - Assignment
Week 9: Multimodal RAG Pipeline with H&M Fashion Dataset

OBJECTIVE: Build a complete multimodal RAG (Retrieval-Augmented Generation) pipeline
that can search through fashion items using both text and image queries, then generate
helpful responses using an LLM.

LEARNING GOALS:
- Understand the three phases of RAG: Retrieval, Augmentation, Generation
- Work with multimodal data (images + text)
- Use vector databases for similarity search
- Integrate LLM for response generation
- Build an end-to-end AI pipeline

DATASET: H&M Fashion Caption Dataset
- 20K+ fashion items with images and text descriptions
- URL: https://huggingface.co/datasets/tomytjandra/h-and-m-fashion-caption

PIPELINE OVERVIEW:
1. RETRIEVAL: Find similar fashion items using vector search
2. AUGMENTATION: Create enhanced prompts with retrieved context
3. GENERATION: Generate helpful responses using LLM

Commands to run:
python assignment_fashion_rag.py --query "black dress for evening"
python assignment_fashion_rag.py --app
"""

import argparse
import os
import re

# Suppress warnings
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Gradio for web interface
import gradio as gr

# Core dependencies
import lancedb
import pandas as pd
import torch
from datasets import load_dataset
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector
from PIL import Image

# LLM dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")


def is_huggingface_space():
    """
    Checks if the code is running within a Hugging Face Spaces environment.

    Returns:
        bool: True if running in HF Spaces, False otherwise.
    """
    if os.environ.get("SYSTEM") == "spaces":
        return True
    else:
        return False


# =============================================================================
# SECTION 1: DATABASE SETUP AND SCHEMA
# =============================================================================


def register_embedding_model(model_name: str = "open-clip") -> Any:
    """
    Register embedding model for vector search

    TODO: Complete this function
    HINT: Use EmbeddingFunctionRegistry to get and create the model

    Args:
        model_name: Name of the embedding model
    Returns:
        Embedding model instance
    """
    # TODO: Get the registry instance
    # registry = ?

    # TODO: Get and create the model
    # model = ?

    # TODO: Return the model
    # return ?

    # DUMMY IMPLEMENTATION - Replace with actual code
    print(f"⚠️ TODO: Register embedding model {model_name}")
    return None


# Global embedding model
clip_model = register_embedding_model()


class FashionItem(LanceModel):
    """
    Schema for fashion items in vector database

    TODO: Complete the schema definition
    HINT: This defines the structure of data stored in the vector database

    REQUIRED FIELDS:
    1. vector: Vector field for CLIP embeddings (use clip_model.ndims())
    2. image_uri: String field for image file paths
    3. description: Optional string field for text descriptions
    """

    # TODO: Add vector field for embeddings
    # vector = ?

    # TODO: Add image field
    # image_uri = ?

    # TODO: Add text description field
    # description = ?

    # DUMMY IMPLEMENTATION - Replace with actual schema
    pass


# =============================================================================
# SECTION 2: RETRIEVAL - Vector Database Operations
# =============================================================================


def setup_fashion_database(
    database_path: str = "fashion_db",
    table_name: str = "fashion_items",
    dataset_name: str = "tomytjandra/h-and-m-fashion-caption",
    sample_size: int = 1000,
    images_dir: str = "fashion_images",
) -> None:
    """
    Set up vector database with H&M fashion dataset

    TODO: Complete this function to:
    1. Connect to LanceDB database
    2. Check if table already exists (skip if it does)
    3. Load H&M dataset from HuggingFace
    4. Process and save images locally
    5. Create vector database table
    """

    # TODO: Connect to LanceDB
    # db = ?

    # TODO: Check if table already exists
    # if table_name in db.table_names():
    #     existing_table = ?            # open table
    #     print(f"✅ Table '{table_name}' already exists with {len(existing_table)} items")
    #     return
    # else:
    #     print(f"🏗️ Table '{table_name}' does not exist, creating new fashion database...")

    # TODO: Load dataset from HuggingFace
    # print("📥 Loading H&M fashion dataset...")
    # dataset = ?
    # train_data = dataset["train"]

    # TODO: Sample data to specified size in the sample_size parameter
    # train_data = ?
    # print(f"Processing {len(train_data)} fashion items...")

    # Create images directory
    os.makedirs(images_dir, exist_ok=True)

    # Process each item
    # table_data = []
    # for i, item in enumerate(train_data):
    #     # Get image and text
    #     image = item["image"]
    #     text = item["text"]

    #     # Save image
    #     image_path = os.path.join(images_dir, f"fashion_{i:04d}.jpg")
    #     image.save(image_path)

    #     # Create record
    #     record = {
    #         "image_uri": image_path,
    #         "description": text
    #     }
    #     table_data.append(record)

    #     if (i + 1) % 100 == 0:
    #         print(f"   Processed {i + 1}/{len(train_data)} items...")

    # TODO: Create vector database table
    # print("🗄️ Creating vector database table...")
    # table = ?
    # print(f"✅ Created table '{table_name}' with {len(table_data)} items")

    # DUMMY IMPLEMENTATION
    print("⚠️ TODO: Implement database setup")
    print(f"Database path: {database_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Sample size: {sample_size}")


def search_fashion_items(
    database_path: str,
    table_name: str,
    query: str,
    search_type: str = "auto",
    limit: int = 3,
) -> Tuple[List[Dict], str]:
    """
    Search for fashion items using text or image query

    TODO: Complete this function to:
    1. Determine if query is text or image (auto-detection)
    2. Connect to the vector database
    3. Perform similarity search using CLIP embeddings
    4. Return search results and detected search type

    STEPS TO IMPLEMENT:
    1. Auto-detect search type: check if query is a file path
    2. Connect to database
    3. Open table
    4. Search based on type:
       - Image: load with PIL and search
       - Text: search directly with string
    5. Return results and search type

    Args:
        database_path: Path to LanceDB database
        table_name: Name of the table to search
        query: Search query (text or image path)
        search_type: "auto", "text", or "image"
        limit: Number of results to return

    Returns:
        Tuple of (results_list, actual_search_type)
    """

    print(f"🔍 Searching for: {query}")

    # TODO: Determine search type automatically
    # HINT: Use os.path.exists(query) to check if query is a file path
    # HINT: If file exists, it's an image search; otherwise, it's text search
    # actual_search_type = ?

    # TODO: Connect to database
    # db = ?

    # TODO: Open the table
    # table = ?

    # TODO: Perform search based on detected type
    # if actual_search_type == "image":
    #     # Load image and search
    #     image = ?
    #     results = ?
    # else:
    #     # Text search
    #     results = ?

    # TODO: Print results found
    # print(f"   Found {len(results)} results using {actual_search_type} search")

    # TODO: Return results and search type
    # return results, actual_search_type

    # DUMMY IMPLEMENTATION
    print("⚠️ TODO: Implement fashion search")
    dummy_results = [
        {
            "description": "solid black jersey top with narrow shoulder straps",
            "image_uri": "fashion_images/fashion_0001.jpg",
        },
        {
            "description": "blue denim jacket with button closure",
            "image_uri": "fashion_images/fashion_0002.jpg",
        },
    ]

    return dummy_results, "text"


# =============================================================================
# SECTION 3: AUGMENTATION - Prompt Engineering
# =============================================================================


def create_fashion_prompt(
    query: str, retrieved_items: List[Dict], search_type: str
) -> str:
    """
    Create enhanced prompt for LLM using retrieved fashion items

    TODO: Complete this function to create a well-structured prompt that:
    1. Creates a system prompt defining the AI assistant's role
    2. Formats retrieved items as context for the LLM
    3. Includes the user's query appropriately
    4. Combines everything into a coherent prompt

    PROMPT STRUCTURE:
    1. System prompt: Define the AI as a fashion assistant
    2. Context section: List retrieved fashion items with descriptions
    3. Query section: Include user's original query
    4. Instruction: Ask for fashion recommendations

    Args:
        query: Original user query
        retrieved_items: List of retrieved fashion items
        search_type: Type of search performed

    Returns:
        Enhanced prompt string for LLM
    """

    # TODO: Create system prompt
    # HINT: Define the AI as a fashion assistant with expertise
    # system_prompt = "You are a ..."

    # TODO: Format retrieved items context
    # context = "Here are some relevant fashion items from our catalog:\n\n"
    # for i, item in enumerate(retrieved_items, 1):
    #     context += f"{i}. {item['description']}\n\n"

    # TODO: Create user query section
    # HINT: Handle different search types (image vs text)
    # if search_type == "image":
    #     query_section = ?
    # else:
    #     query_section = ?

    # TODO: Combine into final prompt
    # HINT: Combine system prompt, context, query section, and response instruction
    # prompt = f"{system_prompt}\n\n{context}\n{query_section}\n\nResponse:"
    # return prompt

    # DUMMY IMPLEMENTATION
    print("⚠️ TODO: Create enhanced prompt")
    return f"Fashion query: {query}\nRetrieved {len(retrieved_items)} items."


# =============================================================================
# SECTION 4: GENERATION - LLM Response Generation
# =============================================================================


def setup_llm_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Tuple[Any, Any]:
    """
    Set up LLM model and tokenizer

    TODO: Complete this function to load the LLM model and tokenizer

    STEPS TO IMPLEMENT:
    1. Load tokenizer
    2. Load model
    3. Configure model settings for GPU/CPU
    5. Return tokenizer and model

    Args:
        model_name: Name of the model to load

    Returns:
        Tuple of (tokenizer, model)
    """

    print(f"🤖 Loading LLM model: {model_name}")

    # TODO: Load tokenizer
    # tokenizer = ?

    # TODO: Load model
    # model = ?

    # TODO: Set pad token if not exists
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # TODO: Print success message and return
    # print("✅ LLM model loaded successfully")
    # return tokenizer, model

    # DUMMY IMPLEMENTATION
    print("⚠️ TODO: Load LLM model and tokenizer")
    return None, None


def generate_fashion_response(
    prompt: str, tokenizer: Any, model: Any, max_tokens: int = 200
) -> str:
    """
    Generate response using LLM

    TODO: Complete this function to generate text using the LLM

    STEPS TO IMPLEMENT:
    1. Check if tokenizer and model are loaded
    2. Encode the prompt with attention mask
    3. Generate response using model.generate()
    4. Decode the response and clean it up
    5. Return the generated text

    Args:
        prompt: Input prompt for the model
        tokenizer: Model tokenizer
        model: LLM model
        max_tokens: Maximum tokens to generate

    Returns:
        Generated response text
    """

    if not tokenizer or not model:
        return "⚠️ LLM not loaded - showing search results only"

    # TODO: Encode prompt with attention mask
    # HINT: Use tokenizer() with return_tensors="pt", truncation=True, max_length=1024, padding=True
    # inputs = ?

    # TODO: Generate response
    # with torch.no_grad():
    #     outputs = model.generate(
    #         inputs.input_ids,
    #         attention_mask=inputs.attention_mask,
    #         max_new_tokens=max_tokens,
    #         temperature=0.7,
    #         do_sample=True,
    #         pad_token_id=tokenizer.eos_token_id,
    #         eos_token_id=tokenizer.eos_token_id
    #     )

    # TODO: Decode response and clean it up
    # full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response = full_response.replace(prompt, "").strip()
    # return response

    # DUMMY IMPLEMENTATION
    print("⚠️ TODO: Generate LLM response")
    return "This is a dummy response. Please implement the LLM generation logic."


# =============================================================================
# SECTION 5: IMAGE STORAGE
# =============================================================================


def save_retrieved_images(
    results: Dict[str, Any], output_dir: str = "retrieved_fashion_images"
) -> List[str]:
    """Save retrieved fashion images to output directory"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    query_safe = re.sub(r"[^\w\s-]", "", str(results["query"]))[:30]
    query_safe = re.sub(r"[-\s]+", "_", query_safe)

    saved_paths = []

    print(f"💾 Saving {len(results['results'])} retrieved images...")

    for i, item in enumerate(results["results"], 1):
        original_path = item["image_uri"]
        image = Image.open(original_path)

        # Generate new filename
        filename = f"{query_safe}_result_{i:02d}.jpg"
        save_path = os.path.join(output_dir, filename)

        # Save image
        image.save(save_path, "JPEG", quality=95)
        saved_paths.append(save_path)

        print(f"   ✅ Saved image {i}: {filename}")
        print(f"      Description: {item.get('description', 'No description')[:60]}...")

    print(f"💾 Saved {len(saved_paths)} images to: {output_dir}")
    return saved_paths


# =============================================================================
# SECTION 6: COMPLETE RAG PIPELINE
# =============================================================================


def run_fashion_rag_pipeline(
    query: str,
    database_path: str = "fashion_db",
    table_name: str = "fashion_items",
    search_type: str = "auto",
    limit: int = 3,
    save_images: bool = True,
) -> Dict[str, Any]:
    """
    Run complete fashion RAG pipeline

    TODO: Complete this function to orchestrate the entire pipeline:
    1. RETRIEVAL: Search for relevant fashion items using vector database
    2. AUGMENTATION: Create enhanced prompt with retrieved context
    3. GENERATION: Generate LLM response using the enhanced prompt
    4. IMAGE STORAGE: Save retrieved images if requested

    This is the main function that ties everything together!

    PIPELINE PHASES:
    Phase 1 - RETRIEVAL: Find similar fashion items
    Phase 2 - AUGMENTATION: Create context-rich prompt
    Phase 3 - GENERATION: Generate helpful response
    Phase 4 - STORAGE: Save retrieved images
    """

    print("🚀 Starting Fashion RAG Pipeline")
    print("=" * 50)

    # PHASE 1: RETRIEVAL
    print("🔍 PHASE 1: RETRIEVAL")
    # TODO: Search for fashion items using the search function
    # HINT: Call search_fashion_items() with the provided parameters
    # results, actual_search_type = ?
    # print(f"   Found {len(results)} relevant items")

    # PHASE 2: AUGMENTATION
    print("📝 PHASE 2: AUGMENTATION")
    # TODO: Create enhanced prompt using retrieved items
    # HINT: Call create_fashion_prompt() with parameters
    # enhanced_prompt = ?
    # print(f"   Created enhanced prompt ({len(enhanced_prompt)} chars)")

    # PHASE 3: GENERATION
    print("🤖 PHASE 3: GENERATION")
    # TODO: Set up LLM and generate response
    # tokenizer, model = ?
    # response = ?
    # print(f"   Generated response ({len(response)} chars)")

    # TODO: Prepare final results dictionary
    # final_results = {
    #     "query": query,
    #     "results": results,
    #     "response": response,
    #     "search_type": actual_search_type
    # }

    # TODO: Save retrieved images if requested
    # if save_images:
    #     saved_image_paths = save_retrieved_images(final_results)
    #     final_results["saved_image_paths"] = saved_image_paths

    # TODO: Return final results
    # return final_results

    # DUMMY IMPLEMENTATION
    print("⚠️ TODO: Implement complete RAG pipeline")

    return {
        "query": query,
        "results": [],
        "response": "Pipeline not implemented yet",
        "search_type": "unknown",
    }


# =============================================================================
# GRADIO WEB APP
# =============================================================================


def fashion_search_app(query):
    """
    Process fashion query and return response with images for Gradio

    TODO: Complete this function to handle web app queries

    STEPS TO IMPLEMENT:
    1. Check if query is provided
    2. Setup database if needed
    3. Run RAG pipeline
    4. Extract LLM response and images
    5. Return formatted results for Gradio
    """

    if not query.strip():
        return "Please enter a search query", []

    # TODO: Setup database if needed (will skip if exists)

    # TODO: Run the RAG pipeline
    # result = ?

    # TODO: Get LLM response
    # llm_response = result['response']

    # TODO: Get retrieved images for display
    # retrieved_images = []
    # for item in result['results']:
    #     if 'image_uri' in item and os.path.exists(item['image_uri']):
    #         img = Image.open(item['image_uri'])
    #         retrieved_images.append(img)

    # TODO: Return response and images
    # return llm_response, retrieved_images

    # DUMMY IMPLEMENTATION
    return "⚠️ TODO: Implement web app functionality", []


def launch_gradio_app():
    """Launch the Gradio web interface"""

    # Create Gradio interface
    with gr.Blocks(title="Fashion RAG Assistant") as app:

        gr.Markdown("# 👗 Fashion RAG Assistant")
        gr.Markdown("Search for fashion items and get AI-powered recommendations!")

        with gr.Row():
            with gr.Column(scale=1):
                # Input
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your fashion query (e.g., 'black dress for evening')",
                    lines=2,
                )

                search_btn = gr.Button("Search", variant="primary")

                # Examples
                gr.Examples(
                    examples=[
                        "black dress for evening",
                        "casual summer outfit",
                        "blue jeans",
                        "white shirt",
                        "winter jacket",
                    ],
                    inputs=query_input,
                )

            with gr.Column(scale=2):
                # Output
                response_output = gr.Textbox(
                    label="Fashion Recommendation", lines=8, interactive=False
                )

        # Retrieved Images
        images_output = gr.Gallery(
            label="Retrieved Fashion Items", columns=3, height=400
        )

        # Connect the search function
        search_btn.click(
            fn=fashion_search_app,
            inputs=query_input,
            outputs=[response_output, images_output],
        )

        # Also trigger on Enter key
        query_input.submit(
            fn=fashion_search_app,
            inputs=query_input,
            outputs=[response_output, images_output],
        )

    print("🚀 Starting Fashion RAG Gradio App...")
    print("📝 Note: First run will download dataset and setup database")
    app.launch(share=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main function to handle command line arguments and run the pipeline"""

    # If running in Hugging Face Spaces, automatically launch the app
    if is_huggingface_space():
        print("🤗 Running in Hugging Face Spaces - launching web app automatically")
        launch_gradio_app()
        return

    parser = argparse.ArgumentParser(
        description="Fashion RAG Pipeline Assignment - SOLUTION"
    )
    parser.add_argument("--query", type=str, help="Search query (text or image path)")
    parser.add_argument("--app", action="store_true", help="Launch Gradio web app")

    args = parser.parse_args()

    # Launch web app if requested
    if args.app:
        launch_gradio_app()
        return

    if not args.query:
        print("❌ Please provide a query with --query or use --app for web interface")
        print("Examples:")
        print("  python solution_fashion_rag.py --query 'black dress for evening'")
        print("  python solution_fashion_rag.py --query 'fashion_images/dress.jpg'")
        print("  python solution_fashion_rag.py --app")
        return

    # Setup database first (will skip if already exists)
    print("🔧 Checking/setting up fashion database...")
    setup_fashion_database()

    # Run the complete RAG pipeline with default settings
    result = run_fashion_rag_pipeline(
        query=args.query,
        database_path="fashion_db",
        table_name="fashion_items",
        search_type="auto",
        limit=3,
        save_images=True,
    )

    # Display results
    print("\n" + "=" * 50)
    print("🎯 PIPELINE RESULTS")
    print("=" * 50)
    print(f"Query: {result['query']}")
    print(f"Search Type: {result['search_type']}")
    print(f"Results Found: {len(result['results'])}")
    print("\n📋 Retrieved Items:")
    for i, item in enumerate(result["results"], 1):
        print(f"{i}. {item.get('description', 'No description')}")

    print(f"\n🤖 LLM Response:")
    print(result["response"])

    # Show saved images info if any
    if result.get("saved_image_paths"):
        print(f"\n📸 Saved Images:")
        for i, path in enumerate(result["saved_image_paths"], 1):
            print(f"{i}. {path}")


if __name__ == "__main__":
    main()