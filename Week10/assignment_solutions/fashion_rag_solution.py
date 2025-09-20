
"""
Fashion RAG Pipeline - SOLUTION
Week 9: Multimodal RAG Pipeline with H&M Fashion Dataset

This is the complete solution for the Fashion RAG Pipeline assignment.
Students should implement the TODO sections in assignment_fashion_rag.py to match this solution.
"""

import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Core dependencies
import lancedb
import pandas as pd
from datasets import load_dataset
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector
from PIL import Image

# LLM dependencies
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Gradio for web interface
import gradio as gr

# Suppress warnings
import warnings
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
    """Register embedding model for vector search"""
    registry = EmbeddingFunctionRegistry.get_instance()
    model = registry.get(model_name).create()
    return model

# Global embedding model
clip_model = register_embedding_model()

class FashionItem(LanceModel):
    """Schema for fashion items in vector database"""

    vector: Vector(clip_model.ndims()) = clip_model.VectorField()
    image_uri: str = clip_model.ImageField()
    description: Optional[str] = None

# =============================================================================
# SECTION 2: RETRIEVAL - Vector Database Operations
# =============================================================================

def setup_fashion_database(
    database_path: str = "fashion_db",
    table_name: str = "fashion_items",
    dataset_name: str = "tomytjandra/h-and-m-fashion-caption",
    sample_size: int = 1000,
    images_dir: str = "fashion_images"
) -> None:
    """Set up vector database with H&M fashion dataset"""
    
    # Connect to LanceDB
    db = lancedb.connect(database_path)
    
    # Check if table already exists
    if table_name in db.table_names():
        existing_table = db.open_table(table_name)
        print(f"✅ Table '{table_name}' already exists with {len(existing_table)} items")
        return
    
    else:
        print(f"🏗️ Table '{table_name}' does not exist, creating new fashion database...")
    
        # Load dataset from HuggingFace
        print("📥 Loading H&M fashion dataset...")
        dataset = load_dataset(dataset_name)
        train_data = dataset["train"]
        
        # Sample data
        train_data = train_data.select(range(sample_size))
        
        print(f"Processing {len(train_data)} fashion items...")
        
        # Create images directory
        os.makedirs(images_dir, exist_ok=True)
        
        # Process each item
        table_data = []
        for i, item in enumerate(train_data):
            # Get image and text
            image = item["image"] 
            text = item["text"]
            
            # Save image
            image_path = os.path.join(images_dir, f"fashion_{i:04d}.jpg")
            image.save(image_path)
            
            # Create record
            record = {
                "image_uri": image_path,
                "description": text
            }
            table_data.append(record)
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(train_data)} items...")
        
        # Create table
        print("🗄️ Creating vector database table...")
        table = db.create_table(table_name, schema=FashionItem, data=table_data, mode="overwrite")
        print(f"✅ Created table '{table_name}' with {len(table_data)} items")

def search_fashion_items(
    database_path: str,
    table_name: str, 
    query: str,
    search_type: str = "auto",
    limit: int = 3
) -> Tuple[List[Dict], str]:
    """Search for fashion items using text or image query"""
    
    print(f"🔍 Searching for: {query}")
    
    # Determine search type
    actual_search_type = "image" if os.path.exists(query) else "text"
    
    # Connect to database
    db = lancedb.connect(database_path)
    table = db.open_table(table_name)
    
    # Perform search based on type
    if actual_search_type == "image":
        image = Image.open(query)
        results = table.search(image).limit(limit).to_list()
    else:
        results = table.search(query).limit(limit).to_list()
    
    print(f"   Found {len(results)} results using {actual_search_type} search")
    return results, actual_search_type

# =============================================================================
# SECTION 3: AUGMENTATION - Prompt Engineering
# =============================================================================

def create_fashion_prompt(query: str, retrieved_items: List[Dict], search_type: str) -> str:
    """Create enhanced prompt for LLM using retrieved fashion items"""
    
    # Create system prompt
    system_prompt = """You are a helpful fashion assistant with expertise in styling and fashion recommendations. 
Based on the provided fashion items, give personalized and practical advice."""
    
    # Format retrieved items context
    context = "Here are some relevant fashion items from our catalog:\n\n"
    for i, item in enumerate(retrieved_items, 1):
        context += f"{i}. {item['description']}\n\n"
    
    # Create user query section
    if search_type == "image":
        query_section = f"Based on the uploaded image and these similar items, provide fashion recommendations."
    else:
        query_section = f"User query: {query}\n\nBased on this request and the similar items above, provide helpful fashion recommendations."
    
    # Combine into final prompt
    prompt = f"{system_prompt}\n\n{context}\n{query_section}\n\nResponse:"
    
    return prompt

# =============================================================================
# SECTION 4: GENERATION - LLM Response Generation  
# =============================================================================

def setup_llm_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Tuple[Any, Any]:
    """Set up LLM model and tokenizer"""
    
    print(f"🤖 Loading LLM model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ LLM model loaded successfully")
    return tokenizer, model

def generate_fashion_response(
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_tokens: int = 200
) -> str:
    """Generate response using LLM"""
    
    if not tokenizer or not model:
        return "⚠️ LLM not loaded - showing search results only"
    
    # Encode prompt with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original prompt from response
    response = full_response.replace(prompt, "").strip()
    
    return response

# =============================================================================
# SECTION 5: IMAGE STORAGE
# =============================================================================

def save_retrieved_images(
    results: Dict[str, Any],
    output_dir: str = "retrieved_fashion_images"
) -> List[str]:
    """Save retrieved fashion images to output directory"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    query_safe = re.sub(r'[^\w\s-]', '', str(results['query']))[:30]
    query_safe = re.sub(r'[-\s]+', '_', query_safe)
    
    saved_paths = []
    
    print(f"💾 Saving {len(results['results'])} retrieved images...")
    
    for i, item in enumerate(results['results'], 1):
        original_path = item['image_uri']
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
    save_images: bool = True
) -> Dict[str, Any]:
    """Run complete fashion RAG pipeline"""
    
    print("🚀 Starting Fashion RAG Pipeline")
    print("=" * 50)
    
    # PHASE 1: RETRIEVAL
    print("🔍 PHASE 1: RETRIEVAL")
    results, actual_search_type = search_fashion_items(
        database_path, table_name, query, search_type, limit
    )
    print(f"   Found {len(results)} relevant items")
    
    # PHASE 2: AUGMENTATION  
    print("📝 PHASE 2: AUGMENTATION")
    enhanced_prompt = create_fashion_prompt(query, results, actual_search_type)
    print(f"   Created enhanced prompt ({len(enhanced_prompt)} chars)")
    
    # PHASE 3: GENERATION
    print("🤖 PHASE 3: GENERATION")
    tokenizer, model = setup_llm_model()
    response = generate_fashion_response(enhanced_prompt, tokenizer, model)
    print(f"   Generated response ({len(response)} chars)")
    
    # Prepare final results
    final_results = {
        "query": query,
        "results": results,
        "response": response,
        "search_type": actual_search_type
    }
    
    # Save retrieved images if requested
    if save_images:
        saved_image_paths = save_retrieved_images(final_results)
        final_results["saved_image_paths"] = saved_image_paths
    
    return final_results

# =============================================================================
# GRADIO WEB APP
# =============================================================================

def fashion_search_app(query):
    """Process fashion query and return response with images for Gradio"""
    
    if not query.strip():
        return "Please enter a search query", []
    
    # Setup database if needed (will skip if exists)
    setup_fashion_database()
    
    # Run the RAG pipeline
    result = run_fashion_rag_pipeline(
        query=query,
        database_path="fashion_db",
        table_name="fashion_items", 
        search_type="auto",
        limit=3,
        save_images=False  # Don't save images for web app
    )
    
    # Get LLM response
    llm_response = result['response']
    
    # Get retrieved images
    retrieved_images = []
    for item in result['results']:
        if 'image_uri' in item and os.path.exists(item['image_uri']):
            img = Image.open(item['image_uri'])
            retrieved_images.append(img)
    
    return llm_response, retrieved_images

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
                    lines=2
                )
                
                search_btn = gr.Button("Search", variant="primary")
                
                # Examples
                gr.Examples(
                    examples=[
                        "black dress for evening",
                        "casual summer outfit", 
                        "blue jeans",
                        "white shirt",
                        "winter jacket"
                    ],
                    inputs=query_input
                )
            
            with gr.Column(scale=2):
                # Output
                response_output = gr.Textbox(
                    label="Fashion Recommendation",
                    lines=8,
                    interactive=False
                )
        
        # Retrieved Images
        images_output = gr.Gallery(
            label="Retrieved Fashion Items",
            columns=3,
            height=400
        )
        
        # Connect the search function
        search_btn.click(
            fn=fashion_search_app,
            inputs=query_input,
            outputs=[response_output, images_output]
        )
        
        # Also trigger on Enter key
        query_input.submit(
            fn=fashion_search_app,
            inputs=query_input,
            outputs=[response_output, images_output]
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
    
    parser = argparse.ArgumentParser(description="Fashion RAG Pipeline Assignment - SOLUTION")
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
        save_images=True
    )
    
    # Display results
    print("\n" + "="*50)
    print("🎯 PIPELINE RESULTS")
    print("="*50)
    print(f"Query: {result['query']}")
    print(f"Search Type: {result['search_type']}")
    print(f"Results Found: {len(result['results'])}")
    print("\n📋 Retrieved Items:")
    for i, item in enumerate(result['results'], 1):
        print(f"{i}. {item.get('description', 'No description')}")
    
    print(f"\n🤖 LLM Response:")
    print(result['response'])
    
    # Show saved images info if any
    if result.get('saved_image_paths'):
        print(f"\n📸 Saved Images:")
        for i, path in enumerate(result['saved_image_paths'], 1):
            print(f"{i}. {path}")

if __name__ == "__main__":
    main() 
