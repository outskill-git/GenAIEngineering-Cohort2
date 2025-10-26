"""
SCRIPT 3/5: generator.py - LLM Setup and Response Generation for Shoe RAG Pipeline

Colab - https://colab.research.google.com/drive/1rq-ywjykHBw7xPXCmd3DmZdK6T9bhDtA?usp=sharing

This script handles the GENERATION phase of the RAG pipeline, including:
- Setting up different LLM providers (Qwen, OpenAI)
- Managing model configurations and parameters
- Generating responses using augmented context
- Handling different model types and API integrations

Key Concepts:
- Large Language Models (LLMs): AI models that generate human-like text
- Model Providers: Different services/frameworks for running LLMs
- API Integration: Connecting to external services like OpenAI
- Local Models: Running models locally with transformers
- Generation Parameters: Temperature, max tokens, etc. for controlling output

Required Dependencies:
- torch: PyTorch for local model inference
- transformers: HuggingFace transformers library
- openai: OpenAI API client (optional)

Commands to run:
# List available models
python generator.py --list-models

# Test Qwen model setup and generation
python generator.py --query "recommend running shoes" --model-provider qwen

# Test OpenAI model generation (requires API key)
python generator.py --query "recommend running shoes" --model-provider openai --openai-api-key YOUR_KEY

# Test with different Qwen model sizes
python generator.py --query "blue sneakers" --model-provider qwen --model-name "Qwen/Qwen2.5-1.5B-Instruct"

# Test generation with custom parameters
python generator.py --query "comfortable shoes" --max-tokens 300 --temperature 0.2

# Test with image search
python generator.py --query "hf_shoe_images/shoe_0000.jpg"

# Test with custom database settings
python generator.py --query "sneakers" --database "myntra_shoes_db" --table-name "myntra_shoes_table"
"""

import argparse
from typing import Dict, List, Optional

import torch
from augmenter import detect_search_type, get_real_shoes_data
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_qwen_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> tuple:
    """GENERATION: Setup Qwen2.5-0.5B model for text generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )
    return tokenizer, model


def setup_openai_client(api_key: str) -> OpenAI:
    """GENERATION: Setup OpenAI client for text generation."""
    if not api_key or api_key.strip() == "":
        raise ValueError("OpenAI API key is required")

    client = OpenAI(api_key=api_key)
    return client


def get_available_models() -> Dict[str, List[str]]:
    """Get available models for each provider."""
    models = {
        "qwen": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    }
    return models


def generate_shoes_rag_response(
    query: str,
    retrieved_shoes: List[Dict[str, any]],
    model_provider: str = "qwen",
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    openai_client: Optional[OpenAI] = None,
    tokenizer=None,
    model=None,
    max_tokens: int = 200,
    use_advanced_prompts: bool = True,
) -> str:
    """GENERATION: Generate RAG response using retrieved shoes context with prompt engineering."""

    if use_advanced_prompts:
        # Use the simplified prompt system
        from augmenter import SimpleShoePrompts

        prompt_manager = SimpleShoePrompts()
        complete_prompt = prompt_manager.generate_prompt(query, retrieved_shoes)
        query_type = prompt_manager.classify_query(query)
        print(f"Using {query_type.value} prompt for query")

    else:
        # Use the basic prompt system (fallback)
        from augmenter import SimpleShoePrompts

        prompt_manager = SimpleShoePrompts()
        context = prompt_manager.format_shoes_context(retrieved_shoes)
        complete_prompt = f"""Based on the following shoe products, answer the user's question:

Shoes:
{context}

Question: {query}

Answer:"""

    if model_provider == "openai":
        if not openai_client:
            raise ValueError("OpenAI client is required for OpenAI models")

        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful shoe recommendation assistant.",
                    },
                    {"role": "user", "content": complete_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    else:  # Qwen model
        if not tokenizer or not model:
            raise ValueError("Tokenizer and model are required for Qwen models")

        inputs = tokenizer(
            complete_prompt, return_tensors="pt", truncation=True, max_length=2048
        )

        # Ensure everything runs on CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Very low temperature to reduce hallucination
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,  # Minimal repetition penalty
                no_repeat_ngram_size=2,
                early_stopping=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return response.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generation component for Shoe RAG Pipeline"
    )
    parser.add_argument("--query", type=str, help="Query for response generation")
    parser.add_argument(
        "--model-provider",
        choices=["qwen", "openai"],
        default="qwen",
        help="Model provider to use",
    )
    parser.add_argument("--model-name", type=str, help="Model name to use")
    parser.add_argument(
        "--openai-api-key", type=str, help="OpenAI API key (required for OpenAI models)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Generation temperature"
    )
    parser.add_argument(
        "--use-advanced-prompts",
        action="store_true",
        default=True,
        help="Use advanced prompt engineering",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--test-setup", action="store_true", help="Test model setup without generation"
    )
    parser.add_argument(
        "--search-type",
        choices=["auto", "text", "image"],
        default="auto",
        help="Search type for retrieving real data",
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

    # List available models
    if args.list_models:
        available_models = get_available_models()
        print("=" * 60)
        print("🤖 AVAILABLE MODELS")
        print("=" * 60)
        for provider, models in available_models.items():
            print(f"\n{provider.upper()} Models:")
            for model in models:
                print(f"  - {model}")
        exit(0)

    # Set default model names
    available_models = get_available_models()
    if not args.model_name:
        args.model_name = available_models[args.model_provider][0]

    print("=" * 60)
    print("🤖 GENERATION SETUP")
    print("=" * 60)
    print(f"Model Provider: {args.model_provider}")
    print(f"Model Name: {args.model_name}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Advanced Prompts: {args.use_advanced_prompts}")

    # Setup models
    tokenizer, model, openai_client = None, None, None

    try:
        if args.model_provider == "openai":
            if not args.openai_api_key:
                print("❌ OpenAI API key is required for OpenAI models")
                exit(1)
            openai_client = setup_openai_client(args.openai_api_key)
            print("✅ OpenAI client setup successful")
        else:
            print(f"🔄 Loading Qwen model: {args.model_name}")
            tokenizer, model = setup_qwen_model(args.model_name)
            print("✅ Qwen model setup successful")
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        exit(1)

    # Test setup only
    if args.test_setup:
        print("✅ Model setup test complete!")
        exit(0)

    # Generate response if query provided
    if args.query:
        print("\n" + "=" * 60)
        print("🔄 GENERATING RESPONSE")
        print("=" * 60)

        # Auto-detect search type if needed
        if args.search_type == "auto":
            detected_search_type = detect_search_type(args.query)
            if detected_search_type == "image":
                print(f"🖼️  Detected image search: {args.query}")
            else:
                print(f"📝 Detected text search: {args.query}")
        else:
            detected_search_type = args.search_type

        # Get real context from retriever
        try:
            real_context = get_real_shoes_data(
                query=args.query,
                search_type=detected_search_type,
                database=args.database,
                table_name=args.table_name,
                limit=args.limit,
            )
            print(f"✅ Retrieved {len(real_context)} real shoes from database")

            # Show retrieved context
            print("\n📊 Retrieved Shoes Context:")
            for i, shoe in enumerate(real_context, 1):
                product_type = shoe.get("product_type", "Shoe")
                gender = shoe.get("gender", "Unisex")
                color = shoe.get("color", "Various colors")
                print(f"  {i}. {product_type} for {gender} ({color})")

        except Exception as e:
            print(f"❌ Failed to retrieve real data: {e}")
            print("Please ensure the database is set up correctly.")
            exit(1)

        try:
            response = generate_shoes_rag_response(
                query=args.query,
                retrieved_shoes=real_context,
                model_provider=args.model_provider,
                model_name=args.model_name,
                openai_client=openai_client,
                tokenizer=tokenizer,
                model=model,
                max_tokens=args.max_tokens,
                use_advanced_prompts=args.use_advanced_prompts,
            )

            print(f"\n🎯 Query: {args.query}")
            print(f"🤖 Response: {response}")
            print("✅ Generation complete!")

        except Exception as e:
            print(f"❌ Generation failed: {e}")

    else:
        print(
            "\n❌ Please provide --query for response generation or --list-models to see available models"
        )
        print("\nExample usage:")
        print("  # Test Qwen generation with real data")
        print("  python generator.py --query 'recommend running shoes'")
        print("  # Test OpenAI generation with real data")
        print(
            "  python generator.py --query 'blue sneakers' --model-provider openai --openai-api-key YOUR_KEY"
        )
        print("  # Test with image search")
        print(
            "  python generator.py --query 'hf_shoe_images/shoe_0000.jpg' --search-type image"
        )
        print("  # List available models")
        print("  python generator.py --list-models")