"""
SCRIPT 5/5: app.py - Gradio Web Interface for Shoe RAG Pipeline

Colab - https://colab.research.google.com/drive/1rq-ywjykHBw7xPXCmd3DmZdK6T9bhDtA?usp=sharing

HuggingFace Spaces - https://huggingface.co/spaces/ishandutta/multimodal-myntra-shoes-rag-pipeline

This script provides a web-based interface for the complete RAG pipeline using Gradio.
It integrates all components (retrieval, augmentation, generation) into a user-friendly web app.

Key Concepts:
- Gradio: Python library for creating web interfaces for ML models
- Web Interface: User-friendly way to interact with RAG pipeline
- Multimodal Input: Supporting both text and image inputs
- Real-time Processing: Live interaction with the RAG system
- Step-by-step Visualization: Showing detailed pipeline execution

Required Dependencies:
- gradio: Web interface framework
- All dependencies from other pipeline components
- Custom CSS for styling

Commands to run:
# Setup database and launch app
python app.py --setup-db

# Launch Gradio web interface
python app.py

# Launch with custom port
python app.py --port 7860

# Launch with public sharing enabled
python app.py --share

# Launch with custom host
python app.py --host 0.0.0.0 --port 8080
"""

import argparse
import os
from typing import Any, Dict, List, Optional

import gradio as gr
from generator import get_available_models
from openai import OpenAI
from rag_pipeline import run_complete_shoes_rag_pipeline_with_details

# Import components from other modules
from retriever import MyntraShoesEnhanced, create_shoes_table_from_hf


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


def gradio_rag_pipeline(
    query,
    image,
    search_type,
    use_advanced_prompts,
    model_provider,
    model_name,
    openai_api_key,
):
    """Gradio interface function for RAG pipeline."""
    try:
        # Validate inputs based on model provider
        if model_provider == "openai":
            if not openai_api_key or openai_api_key.strip() == "":
                return (
                    "❌ OpenAI API key is required for OpenAI models.",
                    "",
                    "",
                    "",
                    "",
                    [],
                )

        # Check if both text and image inputs are provided - this is not allowed
        has_text_input = query and query.strip()
        has_image_input = image is not None

        if has_text_input and has_image_input:
            return (
                "❌ Error: Please provide either a text query OR an image, not both. Choose one input type at a time.",
                "",
                "",
                "",
                "",
                [],
            )

        # Determine the actual query based on inputs
        if search_type == "image" and image is not None:
            actual_query = image
        elif search_type == "text" and query.strip():
            actual_query = query
        elif search_type == "auto":
            if image is not None:
                actual_query = image
            elif query.strip():
                actual_query = query
            else:
                return (
                    "❌ Please provide either a text query or upload an image.",
                    "",
                    "",
                    "",
                    "",
                    [],
                )
        else:
            return (
                "❌ Please provide appropriate input for the selected search type.",
                "",
                "",
                "",
                "",
                [],
            )

        # Run the RAG pipeline with detailed tracking
        rag_result = run_complete_shoes_rag_pipeline_with_details(
            database="myntra_shoes_db",
            table_name="myntra_shoes_table",
            schema=MyntraShoesEnhanced,
            search_query=actual_query,
            limit=3,
            use_llm=True,
            use_advanced_prompts=use_advanced_prompts,
            search_type=search_type,
            model_provider=model_provider,
            model_name=model_name,
            openai_api_key=openai_api_key if model_provider == "openai" else None,
        )

        # Extract detailed step information
        retrieval_details = rag_result.get(
            "retrieval_details", "No retrieval details available"
        )
        augmentation_details = rag_result.get(
            "augmentation_details", "No augmentation details available"
        )
        generation_details = rag_result.get(
            "generation_details", "No generation details available"
        )

        # Format the response
        response = rag_result.get("response", "No response generated")
        search_type_used = rag_result.get("search_type", "unknown")

        # Format results for display
        results_text = f"🔍 Search Type: {search_type_used}\n\n"
        if rag_result.get("prompt_analysis"):
            results_text += (
                f"📝 Query Type: {rag_result['prompt_analysis']['query_type']}\n"
            )
            results_text += (
                f"📊 Results Found: {rag_result['prompt_analysis']['num_results']}\n\n"
            )

        # Prepare image gallery data
        image_gallery = []
        results_details = []

        for i, result in enumerate(rag_result["results"], 1):
            product_type = result.get("product_type", "Shoe")
            gender = result.get("gender", "Unisex")
            color = result.get("color", "Various colors")
            pattern = result.get("pattern", "Standard")
            description = result.get("description", "No description available")
            image_path = result.get("image_path")

            # Add to gallery if image exists
            if image_path and os.path.exists(image_path):
                # Create detailed caption for the image
                caption = f"#{i} - {product_type} for {gender}"
                if color and color not in ["None", None, ""]:
                    caption += f" | Color: {color}"
                if pattern and pattern not in ["None", None, ""]:
                    caption += f" | Pattern: {pattern}"

                image_gallery.append((image_path, caption))

            # Format detailed description
            detail_text = f"**{i}. {product_type} for {gender}**\n"
            detail_text += f"   • Color: {color}\n"
            detail_text += f"   • Pattern: {pattern}\n"
            if description:
                # Show full description without truncation
                detail_text += f"   • Description: {description}\n"
            detail_text += "\n"
            results_details.append(detail_text)

        # Combine all details
        formatted_results = "".join(results_details)

        return (
            response,
            formatted_results,
            retrieval_details,
            augmentation_details,
            generation_details,
            image_gallery,
        )

    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            "",
            "❌ Error occurred",
            "❌ Error occurred",
            "❌ Error occurred",
            [],
        )


def create_gradio_app():
    """Create and launch the Gradio application."""

    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 20px !important;
        background: #f5f5f5;
    }
    .main {
        max-width: 100% !important;
        width: 100% !important;
    }
    /* Header styling */
    .header-section {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-section h1 {
        font-size: 2.5em;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header-section p {
        font-size: 1.2em;
        margin-bottom: 10px;
        opacity: 0.95;
    }
    .output-text {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
    }
    .gallery-container {
        margin-top: 15px;
        width: 100%;
    }
    .search-section {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        height: fit-content;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    .results-section {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        height: fit-content;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .gallery-section {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
        width: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    /* Improve text readability */
    .gradio-textbox textarea {
        background: #fafafa !important;
        border: 1px solid #ddd !important;
        color: #333 !important;
    }
    .gradio-textbox textarea:focus {
        background: #ffffff !important;
        border-color: #667eea !important;
    }
    /* Make gallery images larger */
    .gallery img {
        max-height: 300px !important;
        object-fit: contain !important;
    }
    /* Improve button styling */
    .primary-button {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    .primary-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    /* Section headers */
    .section-header {
        color: #667eea;
        font-weight: bold;
        border-bottom: 2px solid #667eea;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    /* Model settings styling */
    .model-settings {
        background: #f8f9ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #e0e7ff;
    }
    /* API key input styling */
    .api-key-input {
        border: 2px solid #fbbf24 !important;
        background: #fffbeb !important;
    }
    .api-key-input:focus {
        border-color: #f59e0b !important;
        box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.1) !important;
    }
    /* RAG steps styling */
    .rag-steps-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        border: 1px solid #e2e8f0;
    }
    .step-box {
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .step-box:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    /* Different colors for each step */
    .retrieval-step {
        border-left: 4px solid #10b981;
    }
    .augmentation-step {
        border-left: 4px solid #3b82f6;
    }
    .generation-step {
        border-left: 4px solid #8b5cf6;
    }
    """

    with gr.Blocks(css=css, title="👟 Shoe RAG Pipeline") as app:
        # Header Section
        with gr.Row():
            with gr.Column(elem_classes=["header-section"]):
                gr.HTML(
                    """
                <div style="text-align: center;">
                    <h1>👟 Multimodal Shoe RAG Pipeline</h1>
                    <p>This demo showcases a complete <strong>Retrieval-Augmented Generation (RAG)</strong> pipeline for shoe recommendations and search.</p>
                    <div style="display: flex; justify-content: center; gap: 25px; margin-top: 20px; flex-wrap: wrap;">
                        <div>🔍 <strong>Text Search</strong><br/>Natural language queries</div>
                        <div>🖼️ <strong>Image Search</strong><br/>Visual similarity matching</div>
                        <div>🤖 <strong>AI Models</strong><br/>Qwen & OpenAI support</div>
                        <div>🔐 <strong>Secure API</strong><br/>Protected key handling</div>
                        <div>📊 <strong>Structured Results</strong><br/>Detailed product information</div>
                    </div>
                </div>
                """
                )

        with gr.Row(equal_height=False):
            # Left Column - Search Input
            with gr.Column(scale=1, elem_classes=["search-section"]):
                gr.HTML('<h3 class="section-header">🔍 Search Input</h3>')

                query = gr.Textbox(
                    label="Text Query",
                    placeholder="e.g., 'Recommend running shoes for men' or 'Show me casual sneakers'",
                    lines=4,
                    max_lines=6,
                )

                image = gr.Image(
                    label="Upload Shoe Image (for image search)", type="pil", height=220
                )

                with gr.Row():
                    search_type = gr.Radio(
                        choices=["auto", "text", "image"],
                        value="auto",
                        label="Search Type",
                        info="Auto-detect or force specific search type",
                    )

                gr.HTML('<h3 class="section-header">🤖 AI Model Settings</h3>')

                # Model provider selection
                provider_choices = ["qwen", "openai"]

                model_provider = gr.Radio(
                    choices=provider_choices,
                    value="qwen",
                    label="Model Provider",
                    info="Choose between local Qwen models or OpenAI API",
                )

                # Model selection dropdown - will be updated based on provider
                available_models = get_available_models()
                model_name = gr.Dropdown(
                    choices=available_models["qwen"],
                    value="Qwen/Qwen2.5-0.5B-Instruct",
                    label="Model Name",
                    info="Select the specific model to use",
                )

                # OpenAI API Key input (hidden by default)
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="Enter your OpenAI API key (required for OpenAI models)",
                    type="password",
                    visible=False,
                    info="Your API key is secure and not stored",
                )

                use_advanced_prompts = gr.Checkbox(
                    value=True,
                    label="Use Advanced Prompts",
                    info="Enable enhanced prompt engineering for better responses",
                )

                search_btn = gr.Button(
                    "🔍 Search",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-button"],
                )

                # JavaScript to update model choices and show/hide API key based on provider
                def update_model_choices(provider):
                    models = get_available_models()
                    choices = models[provider]
                    default_value = choices[0]
                    api_key_visible = provider == "openai"
                    return gr.Dropdown(
                        choices=choices, value=default_value
                    ), gr.Textbox(visible=api_key_visible)

                model_provider.change(
                    fn=update_model_choices,
                    inputs=[model_provider],
                    outputs=[model_name, openai_api_key],
                )

            # Right Column - Results
            with gr.Column(scale=2, elem_classes=["results-section"]):
                gr.HTML('<h3 class="section-header">🤖 Final AI Response</h3>')
                response_output = gr.Textbox(
                    label="RAG Response",
                    lines=6,
                    max_lines=20,  # Increased max lines
                    elem_classes=["output-text"],
                    show_copy_button=True,
                )

                gr.HTML('<h3 class="section-header">📊 Product Information</h3>')
                results_output = gr.Textbox(
                    label="Retrieved Products Summary",
                    lines=8,
                    max_lines=25,  # Increased max lines
                    elem_classes=["output-text"],
                    show_copy_button=True,
                )

        # Full width section for image gallery
        with gr.Row():
            with gr.Column(elem_classes=["gallery-section"]):
                gr.HTML('<h3 class="section-header">🖼️ Retrieved Shoe Images</h3>')
                image_gallery = gr.Gallery(
                    label="Search Results Gallery",
                    show_label=False,
                    elem_id="gallery",
                    columns=3,
                    rows=1,
                    object_fit="contain",
                    height=350,
                    elem_classes=["gallery-container"],
                    preview=True,
                )

        # RAG Pipeline Steps - Three columns for detailed breakdown
        gr.HTML(
            '<h2 style="text-align: center; color: #667eea; margin: 30px 0 20px 0;">🔍 RAG Pipeline Step-by-Step Breakdown</h2>'
        )

        with gr.Row(equal_height=True, elem_classes=["rag-steps-section"]):
            # Step 1: Retrieval
            with gr.Column(scale=1, elem_classes=["step-box", "retrieval-step"]):
                gr.HTML('<h3 class="section-header">🔍 Step 1: Retrieval</h3>')
                retrieval_output = gr.Textbox(
                    label="Vector Search & Data Retrieval",
                    lines=15,
                    max_lines=30,  # Increased max lines
                    elem_classes=["output-text"],
                    show_copy_button=True,
                    info="Details about vector search, similarity matching, and retrieved products",
                )

            # Step 2: Augmentation
            with gr.Column(scale=1, elem_classes=["step-box", "augmentation-step"]):
                gr.HTML('<h3 class="section-header">📝 Step 2: Augmentation</h3>')
                augmentation_output = gr.Textbox(
                    label="Context Enhancement & Prompt Engineering",
                    lines=15,
                    max_lines=30,  # Increased max lines
                    elem_classes=["output-text"],
                    show_copy_button=True,
                    info="Query analysis, context formatting, and prompt construction",
                )

            # Step 3: Generation
            with gr.Column(scale=1, elem_classes=["step-box", "generation-step"]):
                gr.HTML('<h3 class="section-header">🤖 Step 3: Generation</h3>')
                generation_output = gr.Textbox(
                    label="LLM Response Generation",
                    lines=15,
                    max_lines=30,  # Increased max lines
                    elem_classes=["output-text"],
                    show_copy_button=True,
                    info="Model setup, generation parameters, and response creation",
                )

        # Event handlers
        search_btn.click(
            fn=gradio_rag_pipeline,
            inputs=[
                query,
                image,
                search_type,
                use_advanced_prompts,
                model_provider,
                model_name,
                openai_api_key,
            ],
            outputs=[
                response_output,
                results_output,
                retrieval_output,
                augmentation_output,
                generation_output,
                image_gallery,
            ],
        )

    return app


if __name__ == "__main__":
    # Check if running in Hugging Face Spaces
    if is_huggingface_space():
        print("🤗 Detected Hugging Face Spaces environment")
        print("📋 Using default configuration for HF Spaces...")

        # Use default values for HF Spaces
        class DefaultArgs:
            def __init__(self):
                self.setup_db = True  # Don't setup DB by default in HF Spaces
                self.sample_size = 500
                self.host = "0.0.0.0"  # Use 0.0.0.0 for HF Spaces
                self.port = 7860  # Standard port for HF Spaces
                self.share = False  # HF Spaces handles sharing automatically
                self.debug = False

        args = DefaultArgs()

        print("🔧 HF Spaces Configuration:")
        print(f"   Setup DB: {args.setup_db}")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Share: {args.share}")
        print(f"   Debug: {args.debug}")

    else:
        print("💻 Detected local environment")
        print("📋 Using command line arguments...")

        # Use argparse for local development
        parser = argparse.ArgumentParser(
            description="Gradio Web Interface for Shoe RAG Pipeline"
        )
        parser.add_argument(
            "--setup-db",
            action="store_true",
            help="Setup database from HuggingFace dataset",
        )
        parser.add_argument(
            "--sample-size",
            type=int,
            default=500,
            help="Sample size for database setup",
        )
        parser.add_argument(
            "--host", type=str, default="127.0.0.1", help="Host to run the server on"
        )
        parser.add_argument(
            "--port", type=int, default=7865, help="Port to run the server on"
        )
        parser.add_argument("--share", action="store_true", help="Create a public link")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")

        args = parser.parse_args()

        print("🔧 Local Configuration:")
        print(f"   Setup DB: {args.setup_db}")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Share: {args.share}")
        print(f"   Debug: {args.debug}")

    # Setup database if requested
    if args.setup_db:
        print("🔄 Setting up database from HuggingFace dataset...")
        create_shoes_table_from_hf(
            database="myntra_shoes_db",
            table_name="myntra_shoes_table",
            sample_size=args.sample_size,
            save_images=True,
        )
        print("✅ Database setup complete!")

    print("🚀 Launching Gradio interface...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Share: {args.share}")
    print(f"   Debug: {args.debug}")

    app = create_gradio_app()
    app.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        show_error=args.debug,
    )

    if not is_huggingface_space():
        print("\nExample usage:")
        print("  # Launch with default settings")
        print("  python app.py")
        print("  # Setup database and launch")
        print("  python app.py --setup-db")
        print("  # Launch with public sharing")
        print("  python app.py --share")
        print("  # Launch on custom port")
        print("  python app.py --port 8080")
    else:
        print("\n🤗 Running in Hugging Face Spaces - configuration is automatic!")