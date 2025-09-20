"""
Smart Product Cataloger - Gradio App
Multimodal AI for E-commerce Product Analysis

Google Colab - https://colab.research.google.com/drive/1eFNaidx5TPEhXgzdY9hh7EhDcVZm4GMS?usp=sharing

HuggingFace Spaces - https://huggingface.co/spaces/ishandutta/multimodal-product-cataloger

This app analyzes product images and generates metadata for e-commerce
listings using CLIP and BLIP models.

OVERVIEW:
---------
This application combines zero-shot classification with visual question answering
to analyze product images for e-commerce. It uses:
- CLIP for zero-shot product category classification
- BLIP for image captioning and product description generation
- BLIP VQA for answering specific questions about product attributes

FEATURES:
---------
1. AI-powered product category classification using CLIP
2. Automatic product description generation using BLIP
3. Category-specific attribute extraction via visual Q&A
4. Upload and analyze your own product images
5. Professional e-commerce metadata generation

MODELS USED:
------------
- openai/clip-vit-base-patch32: Zero-shot image classification
- Salesforce/blip-image-captioning-base: Image captioning
- Salesforce/blip-vqa-base: Visual question answering

REQUIREMENTS:
-------------
- torch
- gradio
- transformers
- PIL (Pillow)
- requests

HOW TO RUN:
-----------
1. Install dependencies:
   pip install torch gradio transformers pillow requests

2. Run the application:
   python product_cataloger_app.py

3. Open your browser and navigate to:
   http://localhost:7860

4. Follow the app instructions:
   - Click "Load Models" first (required)
   - Upload product images or use sample URLs
   - Get automatic category classification and metadata

USAGE EXAMPLES:
---------------
Product Categories Supported:
- "clothing" - shirts, dresses, pants, etc.
- "shoes" - sneakers, boots, dress shoes, etc.
- "electronics" - phones, laptops, gadgets, etc.
- "furniture" - chairs, tables, sofas, etc.
- "books" - novels, textbooks, magazines, etc.
- "toys" - games, dolls, educational toys, etc.

The app will automatically classify products and generate relevant
e-commerce metadata including descriptions and category-specific attributes.
"""

import warnings
from typing import Dict, List, Union

import gradio as gr
import requests
import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
    pipeline,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class SmartProductCataloger:
    """
    Main class for analyzing product images and generating e-commerce metadata.

    This class integrates CLIP for classification and BLIP for captioning/VQA
    to create a complete product analysis pipeline for e-commerce applications.

    Attributes:
        device (str): Computing device ('cuda', 'mps', or 'cpu')
        dtype (torch.dtype): Data type for model optimization
        clip_model: CLIP model for zero-shot classification
        clip_processor: CLIP processor for input preprocessing
        blip_caption_model: BLIP model for image captioning
        blip_caption_processor: BLIP processor for captioning
        blip_vqa_model: BLIP model for visual question answering
        blip_vqa_processor: BLIP processor for VQA
        models_loaded (bool): Flag to track if models are loaded
    """

    def __init__(self):
        """Initialize the SmartProductCataloger with device setup and model placeholders."""
        # Automatically detect the best available device for AI computation
        self.device, self.dtype = self.setup_device()

        # Initialize model placeholders - models loaded separately for better UX
        self.clip_model = None  # CLIP classification model
        self.clip_processor = None  # CLIP input processor
        self.blip_caption_model = None  # BLIP captioning model
        self.blip_caption_processor = None  # BLIP captioning processor
        self.blip_vqa_model = None  # BLIP VQA model
        self.blip_vqa_processor = None  # BLIP VQA processor
        self.models_loaded = False  # Track model loading status

    def setup_device(self):
        """
        Setup the optimal computing device and data type for AI models.

        Priority order: CUDA GPU > Apple Silicon MPS > CPU
        Uses float16 for CUDA (memory efficiency) and float32 for others (stability).

        Returns:
            tuple: (device_name, torch_dtype) for model optimization
        """
        if torch.cuda.is_available():
            # NVIDIA GPU available - use CUDA with float16 for memory efficiency
            return "cuda", torch.float16
        elif torch.backends.mps.is_available():
            # Apple Silicon Mac - use Metal Performance Shaders with float32
            return "mps", torch.float32
        else:
            # Fallback to CPU with float32 for compatibility
            return "cpu", torch.float32

    def load_models(self):
        """
        Load all required AI models for product analysis.

        Downloads and initializes:
        1. CLIP for zero-shot product classification
        2. BLIP for image captioning and product descriptions
        3. BLIP VQA for answering specific product attribute questions

        Returns:
            str: Status message indicating success or failure
        """
        # Check if models are already loaded to avoid redundant loading
        if self.models_loaded:
            return "✅ Models already loaded!"

        try:
            print("📦 Loading models...")

            # Load CLIP model for zero-shot product classification
            # Model: openai/clip-vit-base-patch32 (versatile, well-trained model)
            print("📦 Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

            # Load BLIP model for image captioning and product descriptions
            # Model: Salesforce/blip-image-captioning-base (specialized for descriptions)
            print("📦 Loading BLIP caption model...")
            self.blip_caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.blip_caption_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            # Load BLIP VQA model for answering specific product questions
            # Model: Salesforce/blip-vqa-base (specialized for visual Q&A)
            print("📦 Loading BLIP VQA model...")
            self.blip_vqa_model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            )
            self.blip_vqa_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-vqa-base"
            )

            # Set models to evaluation mode for inference (disables dropout, etc.)
            self.blip_caption_model.eval()
            self.blip_vqa_model.eval()

            # Mark models as successfully loaded
            self.models_loaded = True
            return "✅ All models loaded successfully!"

        except Exception as e:
            return f"❌ Error loading models: {str(e)}"

    def load_image_from_url(self, url: str):
        """
        Load an image from a URL with error handling.

        Args:
            url (str): URL of the image to load

        Returns:
            PIL.Image or None: Loaded image in RGB format, or None if failed
        """
        try:
            # Use requests to fetch image data with streaming for efficiency
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise exception for bad status codes

            # Create PIL Image from response and ensure RGB format
            image = Image.open(response.raw).convert("RGB")
            return image

        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None

    def classify_product_image(self, image: Image.Image, candidate_labels: List[str]):
        """
        Classify product image using CLIP zero-shot classification.

        Args:
            image (PIL.Image): Product image to classify
            candidate_labels (List[str]): List of possible product categories

        Returns:
            List[Dict]: Classification results with labels and confidence scores
        """
        if not self.models_loaded:
            return [{"label": "error", "score": 0.0}]

        try:
            print("🔍 Classifying product category...")

            # Use our already-loaded CLIP model directly instead of pipeline
            # Process image and text labels through CLIP processor
            inputs = self.clip_processor(
                text=candidate_labels,  # List of category labels
                images=image,  # PIL Image
                return_tensors="pt",  # Return PyTorch tensors
                padding=True,  # Pad text inputs to same length
            )

            # Get predictions from CLIP model
            with torch.no_grad():  # Disable gradients for inference
                outputs = self.clip_model(**inputs)

                # Calculate probabilities using softmax on logits
                logits_per_image = (
                    outputs.logits_per_image
                )  # Image-text similarity scores
                probs = torch.softmax(
                    logits_per_image, dim=-1
                )  # Convert to probabilities

            # Format results to match pipeline output format
            results = []
            for i, label in enumerate(candidate_labels):
                results.append(
                    {
                        "label": label,
                        "score": float(probs[0][i]),  # Convert tensor to float
                    }
                )

            # Sort by confidence score (highest first) to match pipeline behavior
            results.sort(key=lambda x: x["score"], reverse=True)

            return results

        except Exception as e:
            print(f"❌ Classification error: {e}")
            return [{"label": "error", "score": 0.0}]

    def generate_product_caption(self, image: Image.Image):
        """
        Generate descriptive caption for product image using BLIP.

        Args:
            image (PIL.Image): Product image to describe

        Returns:
            str: Generated product description
        """
        if not self.models_loaded:
            return "❌ Models not loaded."

        try:
            print("📝 Generating product description...")

            # Process image through BLIP captioning processor
            inputs = self.blip_caption_processor(image, return_tensors="pt")

            # Generate caption using BLIP model with beam search for quality
            with torch.no_grad():  # Disable gradients for inference efficiency
                out = self.blip_caption_model.generate(
                    **inputs,
                    max_length=50,  # Maximum description length
                    num_beams=5,  # Beam search for better quality
                    early_stopping=True,  # Stop when end token is generated
                )

            # Decode generated tokens back to readable text
            caption = self.blip_caption_processor.decode(
                out[0], skip_special_tokens=True
            )
            return caption

        except Exception as e:
            return f"❌ Error generating caption: {str(e)}"

    def ask_about_product(self, image: Image.Image, question: str):
        """
        Answer specific questions about product using BLIP Visual Question Answering.

        Args:
            image (PIL.Image): Product image to analyze
            question (str): Question to ask about the product

        Returns:
            str: Answer to the question or error message
        """
        if not self.models_loaded:
            return "❌ Models not loaded."

        try:
            # Process both image and question together through BLIP VQA processor
            inputs = self.blip_vqa_processor(image, question, return_tensors="pt")

            # Generate answer using BLIP VQA model
            with torch.no_grad():  # Disable gradients for inference
                out = self.blip_vqa_model.generate(
                    **inputs,
                    max_length=20,  # Answers are typically short
                    num_beams=5,  # Beam search for better quality
                    early_stopping=True,  # Stop when end token is generated
                )

            # Decode generated tokens to get the final answer
            answer = self.blip_vqa_processor.decode(out[0], skip_special_tokens=True)
            return answer.strip()  # Remove extra whitespace

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_category_questions(self, category: str):
        """
        Get relevant questions for specific product categories.

        Each category has tailored questions to extract the most useful
        e-commerce metadata and product attributes.

        Args:
            category (str): Product category name

        Returns:
            List[str]: List of relevant questions for the category
        """
        # Comprehensive mapping of categories to relevant e-commerce questions
        question_map = {
            "shoes": [
                "What color are these shoes?",
                "What type of shoes are these?",
                "What brand are these shoes?",
                "What material are these shoes made of?",
                "Are these sneakers?",
            ],
            "clothing": [
                "What color is this clothing?",
                "What type of clothing is this?",
                "What material is this clothing made of?",
                "What size is this clothing?",
                "Is this formal or casual wear?",
            ],
            "electronics": [
                "What type of device is this?",
                "What brand is this device?",
                "What color is this device?",
                "Is this a smartphone or tablet?",
                "Does this have a screen?",
            ],
            "furniture": [
                "What type of furniture is this?",
                "What color is this furniture?",
                "What material is this furniture made of?",
                "Is this indoor or outdoor furniture?",
                "How many people can use this?",
            ],
            "books": [
                "What type of book is this?",
                "What color is the book cover?",
                "Is this a hardcover or paperback?",
                "Does this book have text on the cover?",
                "Is this a fiction or non-fiction book?",
            ],
            "toys": [
                "What type of toy is this?",
                "What color is this toy?",
                "Is this toy for children or adults?",
                "What material is this toy made of?",
                "Is this an educational toy?",
            ],
        }

        # Return category-specific questions or default generic questions
        return question_map.get(
            category,
            [
                "What color is this?",
                "What type of item is this?",
                "What is this made of?",
            ],
        )

    def analyze_product_complete(self, image: Image.Image):
        """
        Complete end-to-end product analysis pipeline.

        This method combines all analysis steps:
        1. Classify product category using CLIP
        2. Generate product description using BLIP
        3. Ask category-specific questions using BLIP VQA
        4. Compile results into structured e-commerce metadata

        Args:
            image (PIL.Image): Product image to analyze

        Returns:
            Dict: Complete analysis results with category, description, and attributes
        """
        if not self.models_loaded:
            return {"error": "Models not loaded", "status": "failed"}

        if image is None:
            return {"error": "No image provided", "status": "failed"}

        try:
            print("🚀 Starting complete product analysis...")

            # Step 1: Classify product category using CLIP zero-shot classification
            product_categories = [
                "clothing",
                "shoes",
                "electronics",
                "furniture",
                "books",
                "toys",
            ]
            classification_results = self.classify_product_image(
                image, product_categories
            )

            if classification_results[0]["label"] == "error":
                return {"error": "Classification failed", "status": "failed"}

            top_category = classification_results[0]  # Highest confidence category

            # Step 2: Generate product description using BLIP captioning
            description = self.generate_product_caption(image)

            # Step 3: Get category-specific questions and ask them using VQA
            category = top_category["label"]
            questions = self.get_category_questions(category)

            # Ask each question and collect answers for product attributes
            qa_results = {}
            for question in questions:
                answer = self.ask_about_product(image, question)
                qa_results[question] = answer

            # Step 4: Compile everything into structured e-commerce metadata
            result = {
                "category": {"name": category, "confidence": top_category["score"]},
                "description": description,
                "attributes": qa_results,
                "all_categories": classification_results,  # Include all classification results
                "status": "success",
            }

            return result

        except Exception as e:
            return {"error": str(e), "status": "failed"}


# Initialize the main product cataloger instance
# This creates a single instance used throughout the app
product_cataloger = SmartProductCataloger()

# Define Gradio interface wrapper functions
# These functions adapt the class methods for use with Gradio components


def load_models_interface():
    """
    Gradio interface wrapper for loading AI models.

    Returns:
        str: Status message from model loading process
    """
    return product_cataloger.load_models()


def analyze_upload_interface(image):
    """
    Gradio interface wrapper for analyzing directly uploaded product images.

    Args:
        image (PIL.Image or None): Image uploaded through Gradio interface

    Returns:
        tuple: (image, analysis_text, category_text, attributes_text) for Gradio outputs
    """
    # Validate image input from Gradio component
    if image is None:
        error_msg = "❌ Please upload a product image."
        return None, error_msg, error_msg, error_msg

    # Run complete analysis pipeline on the uploaded image
    result = product_cataloger.analyze_product_complete(image)

    if result.get("status") == "failed":
        error_msg = f"❌ Analysis failed: {result.get('error', 'Unknown error')}"
        return image, error_msg, error_msg, error_msg

    # Format results for display in Gradio interface
    # Main analysis summary
    analysis_text = f"""🔍 PRODUCT ANALYSIS COMPLETE

📝 Description: {result['description']}

🏷️ Category: {result['category']['name']} (confidence: {result['category']['confidence']:.3f})

✅ Analysis Status: {result['status']}"""

    # Category classification results
    category_text = "🏷️ CATEGORY CLASSIFICATION\n\n"
    for cat in result["all_categories"]:
        category_text += f"• {cat['label']}: {cat['score']:.3f}\n"

    # Product attributes from VQA
    attributes_text = "🔍 PRODUCT ATTRIBUTES\n\n"
    for question, answer in result["attributes"].items():
        attributes_text += f"❓ {question}\n💡 {answer}\n\n"

    # Return the same image for display along with analysis results
    return image, analysis_text, category_text, attributes_text


def analyze_url_interface(url):
    """
    Gradio interface wrapper for analyzing product from URL.

    Args:
        url (str): Image URL from Gradio textbox

    Returns:
        tuple: (image, analysis_text, category_text, attributes_text) for Gradio outputs
    """
    # Validate URL input
    if not url or not url.strip():
        error_msg = "❌ Please provide an image URL."
        return None, error_msg, error_msg, error_msg

    # Load image from URL
    image = product_cataloger.load_image_from_url(url.strip())
    if image is None:
        error_msg = "❌ Failed to load image from URL. Please check the URL."
        return None, error_msg, error_msg, error_msg

    # Run complete analysis pipeline on the loaded image
    result = product_cataloger.analyze_product_complete(image)

    if result.get("status") == "failed":
        error_msg = f"❌ Analysis failed: {result.get('error', 'Unknown error')}"
        return image, error_msg, error_msg, error_msg

    # Format results for display in Gradio interface
    # Main analysis summary
    analysis_text = f"""🔍 PRODUCT ANALYSIS COMPLETE

📝 Description: {result['description']}

🏷️ Category: {result['category']['name']} (confidence: {result['category']['confidence']:.3f})

✅ Analysis Status: {result['status']}"""

    # Category classification results
    category_text = "🏷️ CATEGORY CLASSIFICATION\n\n"
    for cat in result["all_categories"]:
        category_text += f"• {cat['label']}: {cat['score']:.3f}\n"

    # Product attributes from VQA
    attributes_text = "🔍 PRODUCT ATTRIBUTES\n\n"
    for question, answer in result["attributes"].items():
        attributes_text += f"❓ {question}\n💡 {answer}\n\n"

    return image, analysis_text, category_text, attributes_text


# Create Gradio interface using Blocks for custom layout
# gr.Blocks: Allows custom layout with rows, columns, and advanced components
# title: Sets the browser tab title for the web interface
with gr.Blocks(title="Smart Product Cataloger") as app:
    # gr.Markdown: Renders markdown text with formatting, emojis, and styling
    # Supports HTML-like formatting for headers, lists, bold text, etc.
    gr.Markdown(
        """
    # 🛍️ Smart Product Cataloger
    
    **Multimodal AI for E-commerce Product Analysis**
    
    This app analyzes product images and generates metadata for e-commerce listings
    using CLIP for classification and BLIP for captioning and visual question answering.
    
    ## 🚀 How to use:
    1. **Load Models** - Click to load the AI models (required first step)
    2. **Upload Image** - Upload a product image directly for analysis
    3. **URL Analysis** - Analyze products from image URLs
    """
    )

    # Model loading section
    # gr.Row: Creates horizontal layout container for organizing components side by side
    with gr.Row():
        # gr.Column: Creates vertical layout container within the row
        with gr.Column():
            # Markdown for section header with emoji and formatting
            gr.Markdown("### 📦 Step 1: Load Models")

            # gr.Button: Interactive button component
            # variant="primary": Makes button blue/prominent (primary action)
            # size="lg": Large button size for better visibility
            load_btn = gr.Button("🔄 Load Models", variant="primary", size="lg")

            # gr.Textbox: Text input/output component
            # label: Display label above the textbox
            # interactive=False: Makes textbox read-only (output only)
            load_status = gr.Textbox(label="Status", interactive=False)

    # Event handler: Connects button click to function
    # fn: Function to call when button is clicked
    # outputs: Which component(s) receive the function's return value
    load_btn.click(
        fn=load_models_interface,  # Function to execute
        outputs=load_status,  # Component to update with result
    )

    # Markdown horizontal rule for visual separation between sections
    gr.Markdown("---")

    # Direct image upload section
    with gr.Row():
        # Left column for image upload and controls
        # scale=1: Equal width columns (both columns take same space)
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Step 2: Upload Product Image")

            # gr.Image for file upload functionality
            # When no image is provided, shows upload interface
            # label: Text shown above upload area
            # height: Fixed pixel height for consistent layout
            uploaded_image = gr.Image(label="Upload Product Image", height=400)

            # Primary action button for direct image analysis
            # variant="primary": Blue/prominent styling for main action
            upload_analyze_btn = gr.Button(
                "🔍 Analyze Uploaded Image", variant="primary"
            )

        # Right column for displaying the uploaded image
        with gr.Column(scale=1):
            # gr.Image: Component for displaying the uploaded image
            # label: Caption shown above image
            # height: Consistent sizing with upload area
            upload_image_display = gr.Image(label="Uploaded Image", height=400)

    # Upload analysis results section with three columns for different result types
    with gr.Row():
        # Column for main analysis summary
        with gr.Column(scale=1):
            # Multi-line textbox for displaying main analysis results
            # lines=8: Adequate height for analysis summary
            # interactive=False: Read-only output field
            upload_analysis_output = gr.Textbox(
                label="📋 Analysis Summary", lines=8, interactive=False
            )

        # Column for category classification results
        with gr.Column(scale=1):
            # Output textbox for category classification scores
            upload_category_output = gr.Textbox(
                label="🏷️ Category Classification", lines=8, interactive=False
            )

        # Column for product attributes from VQA
        with gr.Column(scale=1):
            # Output textbox for detailed product attributes
            upload_attributes_output = gr.Textbox(
                label="🔍 Product Attributes", lines=8, interactive=False
            )

    # Event handler for upload analyze button
    # inputs: Component whose value is passed to function
    # outputs: Components that receive function return values (order matters)
    upload_analyze_btn.click(
        fn=analyze_upload_interface,  # Function to call
        inputs=uploaded_image,  # Input component
        outputs=[
            upload_image_display,
            upload_analysis_output,
            upload_category_output,
            upload_attributes_output,
        ],  # Output components
    )

    # Visual separator between sections
    gr.Markdown("---")

    # URL analysis section for analyzing products from web URLs
    with gr.Row():
        # Left column for URL input
        with gr.Column(scale=1):
            gr.Markdown("### 🌐 Step 3: Analyze from URL")

            # gr.Textbox for URL input
            # label: Text shown above input field
            # placeholder: Hint text shown when field is empty
            # lines=1: Single line input for URLs
            url_input = gr.Textbox(
                label="Product Image URL",
                placeholder="https://example.com/product-image.jpg",
                lines=1,
            )

            # Secondary action button for URL analysis
            # variant="secondary": Gray/muted styling (less prominent than primary)
            url_analyze_btn = gr.Button("🔗 Analyze from URL", variant="secondary")

        # Right column for URL-loaded image display
        with gr.Column(scale=1):
            # Image component to show the loaded image from URL
            url_image_display = gr.Image(label="Loaded Image", height=400)

    # URL analysis results section with three columns for different result types
    with gr.Row():
        # Three columns for different types of analysis results
        with gr.Column(scale=1):
            # Main analysis results for URL-loaded image
            url_analysis_output = gr.Textbox(
                label="📋 Analysis Summary", lines=8, interactive=False
            )

        with gr.Column(scale=1):
            # Category classification for URL-loaded image
            url_category_output = gr.Textbox(
                label="🏷️ Category Classification", lines=8, interactive=False
            )

        with gr.Column(scale=1):
            # Product attributes for URL-loaded image
            url_attributes_output = gr.Textbox(
                label="🔍 Product Attributes", lines=8, interactive=False
            )

    # Event handler for URL analysis button
    # inputs: URL textbox component
    # outputs: All four components (image + three analysis results)
    url_analyze_btn.click(
        fn=analyze_url_interface,  # Function to execute
        inputs=url_input,  # Input component
        outputs=[
            url_image_display,
            url_analysis_output,
            url_category_output,
            url_attributes_output,
        ],  # Output components
    )

    # Final section with examples and usage tips
    # Triple-quoted string allows multi-line markdown content
    gr.Markdown(
        """
    ---
    ### 📝 Example Product Categories:
    - **Clothing**: shirts, dresses, pants, jackets
    - **Shoes**: sneakers, boots, dress shoes, sandals
    - **Electronics**: phones, laptops, headphones, tablets
    - **Furniture**: chairs, tables, sofas, desks
    - **Books**: novels, textbooks, magazines, comics
    - **Toys**: games, dolls, educational toys, puzzles
    
    ### 🔗 Sample Product URLs:
    - Shoes: https://images.unsplash.com/photo-1542291026-7eec264c27ff
    - Electronics: https://images.unsplash.com/photo-1511707171634-5f897ff02aa9
    - Clothing: https://images.unsplash.com/photo-1521572163474-6864f9cf17ab
    
    """
    )

if __name__ == "__main__":
    """
    Launch the Gradio app when script is run directly.

    Configuration:
        server_name="0.0.0.0": Allow access from any IP address
        server_port=7860: Use port 7860 (Gradio default)
        share=True: Create public Gradio link for sharing
        debug=True: Enable debug mode for development
    """
    app.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860,  # Standard Gradio port
        share=True,  # Generate shareable public link
        debug=True,  # Enable debug logging
    )