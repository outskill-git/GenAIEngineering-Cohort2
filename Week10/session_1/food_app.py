"""
Smart Food Image Generator - Gradio App
Multimodal AI for Food Delivery Business

Google Colab - https://colab.research.google.com/drive/1GKJUa7zI9Ei0IkgNzRlBEhijdILhdymc?usp=sharing

HuggingFace Spaces - https://huggingface.co/spaces/ishandutta/multimodal-food-image-generation-and-analysis

This app generates food images and analyzes them for safety
using Stable Diffusion and BLIP models.

OVERVIEW:
---------
This application combines text-to-image generation with visual question answering
to create and analyze food images. It uses:
- Stable Diffusion v1.5 for generating high-quality food images from text descriptions
- BLIP VQA (Visual Question Answering) for analyzing food safety, ingredients, and allergens

FEATURES:
---------
1. AI-powered food image generation from text descriptions
2. Automatic food analysis including allergen detection
3. Upload and analyze your own food images
4. Professional food photography style generation

MODELS USED:
------------
- runwayml/stable-diffusion-v1-5: Text-to-image generation
- Salesforce/blip-vqa-base: Visual question answering for food analysis

REQUIREMENTS:
-------------
- torch
- gradio
- diffusers
- transformers
- PIL (Pillow)

HOW TO RUN:
-----------
1. Install dependencies:
   pip install torch gradio diffusers transformers pillow

2. Run the application:
   python food_app.py

3. Open your browser and navigate to:
   http://localhost:7860

4. Follow the app instructions:
   - Click "Load Models" first (required)
   - Generate food images with descriptions
   - Analyze food safety and allergens

USAGE EXAMPLES:
---------------
Food Descriptions:
- "butter chicken with rice"
- "chocolate chip cookies"
- "grilled fish with vegetables"
- "veg margherita pizza"

The app will generate professional-looking food images and automatically
analyze them for allergens, dietary restrictions, and safety information.
"""

import warnings

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class SmartFoodGenerator:
    """
    Main class for generating and analyzing food images using AI models.

    This class integrates Stable Diffusion for image generation and BLIP for
    visual question answering to create a complete food analysis pipeline.

    Attributes:
        device (str): Computing device ('cuda', 'mps', or 'cpu')
        dtype (torch.dtype): Data type for model optimization
        text2img_pipe: Stable Diffusion pipeline for image generation
        blip_model: BLIP model for visual question answering
        blip_processor: BLIP processor for input preprocessing
        models_loaded (bool): Flag to track if models are loaded
    """

    def __init__(self):
        """Initialize the SmartFoodGenerator with device setup and model placeholders."""
        # Automatically detect the best available device for AI computation
        self.device, self.dtype = self.setup_device()

        # Initialize model placeholders - models loaded separately for better UX
        self.text2img_pipe = None  # Stable Diffusion pipeline
        self.blip_model = None  # BLIP VQA model
        self.blip_processor = None  # BLIP input processor
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
        Load all required AI models for food generation and analysis.

        Downloads and initializes:
        1. Stable Diffusion v1.5 for text-to-image generation
        2. BLIP VQA for visual question answering and food analysis

        Returns:
            str: Status message indicating success or failure
        """
        # Check if models are already loaded to avoid redundant loading
        if self.models_loaded:
            return "✅ Models already loaded!"

        try:
            print("📦 Loading models...")

            # Load Stable Diffusion pipeline for food image generation
            # Model: runwayml/stable-diffusion-v1-5 (popular, well-trained model)
            self.text2img_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",  # Pre-trained model identifier
                torch_dtype=self.dtype,  # Use optimized data type
                safety_checker=None,  # Disable safety checker for food images
                requires_safety_checker=False,  # Skip safety requirements
            )
            # Move pipeline to the optimal device (GPU/MPS/CPU)
            self.text2img_pipe = self.text2img_pipe.to(self.device)

            # Load BLIP model for visual question answering and food analysis
            # Model: Salesforce/blip-vqa-base (specialized for visual Q&A)
            self.blip_model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            )
            self.blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-vqa-base"
            )

            # Set model to evaluation mode for inference (disables dropout, etc.)
            self.blip_model.eval()

            # Mark models as successfully loaded
            self.models_loaded = True
            return "✅ All models loaded successfully!"

        except Exception as e:
            return f"❌ Error loading models: {str(e)}"

    def generate_food_image(self, food_description, seed=42):
        """
        Generate professional food image from text description using Stable Diffusion.

        Args:
            food_description (str): Text description of the food to generate
            seed (int): Random seed for reproducible results (default: 42)

        Returns:
            tuple: (PIL.Image or None, status_message)
        """
        # Validate that models are loaded before proceeding
        if not self.models_loaded:
            return None, "❌ Models not loaded. Please load models first."

        # Validate input description
        if not food_description:
            return None, "❌ Please provide a food description."

        try:
            print(f"🍽️ Generating: {food_description}")

            # Enhanced prompt engineering for better food photography results
            # Adds professional photography keywords to improve image quality
            prompt = f"{food_description}, professional food photography, appetizing, restaurant style"

            # Set random seed for reproducible results across runs
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)  # Also set CUDA seed if using GPU

            # Generate image using Stable Diffusion pipeline
            with torch.no_grad():  # Disable gradient computation for inference
                result = self.text2img_pipe(
                    prompt=prompt,  # Main text prompt
                    negative_prompt="blurry, low quality, unappetizing",  # What to avoid
                    num_inference_steps=20,  # Number of denoising steps
                    guidance_scale=7.5,  # How closely to follow prompt
                    height=512,  # Output image height
                    width=512,  # Output image width
                )

            return result.images[0], "✅ Food image generated successfully!"

        except Exception as e:
            return None, f"❌ Error generating image: {str(e)}"

    def ask_about_food(self, image, question):
        """
        Ask questions about food using BLIP Visual Question Answering.

        Args:
            image (PIL.Image): Food image to analyze
            question (str): Question to ask about the image

        Returns:
            str: Answer to the question or error message
        """
        # Validate that models are loaded
        if not self.models_loaded:
            return "❌ Models not loaded."

        try:
            # Process image and question through BLIP processor
            # Converts image and text to model-compatible tensors
            inputs = self.blip_processor(image, question, return_tensors="pt")

            # Generate answer using BLIP VQA model
            with torch.no_grad():  # Disable gradients for inference
                out = self.blip_model.generate(
                    **inputs,  # Input tensors (image + question)
                    max_length=200,  # Maximum response length
                    num_beams=5,  # Beam search for better quality answers
                )

            # Decode the generated tokens back to text
            answer = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return answer.strip()  # Remove extra whitespace

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def analyze_food_safety(self, food_image):
        """
        Comprehensive food analysis including allergens and dietary information.

        Uses BLIP VQA to analyze the food image for:
        - General description
        - Common allergens (dairy, nuts, eggs, gluten)
        - Dietary restrictions (vegetarian, spicy)

        Args:
            food_image (PIL.Image): Food image to analyze

        Returns:
            str: Formatted analysis results or error message
        """
        # Validate that models are loaded
        if not self.models_loaded:
            return "❌ Models not loaded."

        # Validate that image is provided
        if food_image is None:
            return "❌ No image provided."

        try:
            print("🔬 Analyzing food ...")

            # Get basic description of the food using VQA
            description = self.ask_about_food(food_image, "Describe the food")

            # Define common allergen questions for systematic checking
            # These cover the most common food allergens
            allergen_questions = [
                "Does this contain dairy or milk?",  # Dairy products
                "Does this contain nuts?",  # Tree nuts/peanuts
                "Does this contain eggs?",  # Egg products
                "Does this contain wheat or gluten?",  # Gluten-containing grains
            ]

            # Check each allergen by asking specific questions
            allergens = []
            for question in allergen_questions:
                answer = self.ask_about_food(food_image, question)
                # If the answer contains "yes", extract the allergen name
                if "yes" in answer.lower():
                    # Extract allergen name from question (e.g., "dairy or milk" from question)
                    allergen = question.split("contain ")[-1].split("?")[0]
                    allergens.append(allergen)

            # Get additional dietary information
            vegetarian = self.ask_about_food(food_image, "Is this vegetarian?")
            spicy = self.ask_about_food(food_image, "Is this spicy?")

            # Format comprehensive analysis results with emojis for better UX
            analysis_text = f"🔬 FOOD SAFETY ANALYSIS\n\n"
            analysis_text += f"📝 Description: {description}\n\n"
            analysis_text += f"⚠️  Allergens: {', '.join(allergens) if allergens else 'None detected'}\n\n"
            analysis_text += f"🥬 Vegetarian: {vegetarian}\n\n"
            analysis_text += f"🌶️  Spicy: {spicy}"

            return analysis_text

        except Exception as e:
            return f"❌ Error analyzing food: {str(e)}"

    def generate_and_analyze_food(self, food_description, seed=42):
        """
        Complete end-to-end pipeline: generate food image and analyze it.

        This method combines image generation and analysis into a single workflow:
        1. Generate professional food image from text description
        2. Automatically analyze the generated image for allergens

        Args:
            food_description (str): Text description of food to generate
            seed (int): Random seed for reproducible image generation

        Returns:
            tuple: (PIL.Image or None, analysis_text or error_message)
        """
        # Validate that models are loaded
        if not self.models_loaded:
            return None, "❌ Models not loaded. Please load models first."

        # Validate input description
        if not food_description:
            return None, "❌ Please provide a food description."

        try:
            print(f"🚀 Complete pipeline for: {food_description}")

            # Step 1: Generate food image using Stable Diffusion
            food_image, gen_status = self.generate_food_image(food_description, seed)

            # Check if image generation was successful
            if food_image is None:
                return None, gen_status

            # Step 2: Analyze the generated food image for safety
            analysis = self.analyze_food_safety(food_image)

            return food_image, analysis

        except Exception as e:
            return None, f"❌ Error in pipeline: {str(e)}"


# Initialize the main food generator instance
# This creates a single instance that will be used throughout the app
food_generator = SmartFoodGenerator()

# Define Gradio interface wrapper functions
# These functions adapt the class methods for use with Gradio components


def load_models_interface():
    """
    Gradio interface wrapper for loading AI models.

    Returns:
        str: Status message from model loading process
    """
    return food_generator.load_models()


def generate_food_interface(food_description, seed):
    """
    Gradio interface wrapper for generating and analyzing food images.

    Args:
        food_description (str): User input food description from Gradio textbox
        seed (int): Random seed value from Gradio slider

    Returns:
        tuple: (generated_image, analysis_text) for Gradio outputs
    """
    image, status = food_generator.generate_and_analyze_food(food_description, seed)
    return image, status


def analyze_uploaded_food(image):
    """
    Gradio interface wrapper for analyzing uploaded food images.

    Args:
        image (PIL.Image or None): Image uploaded through Gradio interface

    Returns:
        str: Food safety analysis results or error message
    """
    # Validate image input from Gradio component
    if image is None:
        return "❌ Please upload an image."
    return food_generator.analyze_food_safety(image)


def ask_question_interface(image, question):
    """
    Gradio interface wrapper for asking questions about food images.

    Args:
        image (PIL.Image or None): Image from Gradio component
        question (str): Question text from Gradio textbox

    Returns:
        str: Answer to the question or error message
    """
    # Validate inputs from Gradio components
    if image is None:
        return "❌ Please upload an image."
    if not question:
        return "❌ Please enter a question."
    return food_generator.ask_about_food(image, question)


# Create Gradio interface using Blocks for custom layout
# gr.Blocks: Allows custom layout with rows, columns, and advanced components
# title: Sets the browser tab title for the web interface
with gr.Blocks(title="Smart Food Image Generator") as app:
    # gr.Markdown: Renders markdown text with formatting, emojis, and styling
    # Supports HTML-like formatting for headers, lists, bold text, etc.
    gr.Markdown(
        """
    # 🍽️ Smart Food Image Generator
    
    **Multimodal AI for Food Delivery Business**
    
    This app generates professional food images from text descriptions and analyzes them for safety, 
    ingredients, and allergens using Stable Diffusion and BLIP models.
    
    ## 🚀 How to use:
    1. **Load Models** - Click to load the AI models (required first step)
    2. **Generate Food** - Enter a food description to generate and analyze images
    3. **Analyze Food** - Upload your own food images for safety analysis
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

    # Food generation section with two-column layout
    with gr.Row():
        # Left column for inputs
        # scale=1: Equal width columns (both columns take same space)
        with gr.Column(scale=1):
            gr.Markdown("### 🍽️ Step 2: Generate Food Images")

            # gr.Textbox for multi-line text input
            # label: Text shown above input field
            # placeholder: Hint text shown when field is empty
            # lines=2: Makes textbox 2 lines tall for better UX
            food_input = gr.Textbox(
                label="Food Description",
                placeholder="e.g., butter chicken with rice, chocolate chip cookies, grilled salmon with vegetables",
                lines=2,
            )

            # gr.Slider: Numeric input with draggable slider interface
            # minimum/maximum: Range bounds for the slider
            # value: Default/initial value
            # step: Increment size when dragging slider
            seed_input = gr.Slider(
                label="Seed (for reproducible results)",
                minimum=1,  # Lowest possible value
                maximum=1000,  # Highest possible value
                value=42,  # Default value (common ML seed)
                step=1,  # Integer increments
            )

            # Primary action button for main workflow
            generate_btn = gr.Button("🎨 Generate & Analyze Food", variant="primary")

        # Right column for outputs (equal width due to scale=1)
        with gr.Column(scale=1):
            # gr.Image: Component for displaying images
            # label: Caption shown above image
            # height: Fixed pixel height for consistent layout
            generated_image = gr.Image(label="Generated Food Image", height=400)

            # Multi-line textbox for displaying analysis results
            # lines=10: Tall textbox to show complete analysis
            # interactive=False: Read-only output field
            analysis_output = gr.Textbox(
                label="Food Analysis", lines=10, interactive=False
            )

    # Event handler for generation button
    # inputs: List of components whose values are passed to function
    # outputs: List of components that receive function return values
    generate_btn.click(
        fn=generate_food_interface,  # Function to call
        inputs=[food_input, seed_input],  # Input components (order matters)
        outputs=[generated_image, analysis_output],  # Output components (order matters)
    )

    # Visual separator between sections
    gr.Markdown("---")

    # Food analysis section for user-uploaded images
    with gr.Row():
        # Left column for image upload and controls
        with gr.Column(scale=1):
            gr.Markdown("### 🔬 Step 3: Analyze Your Food Images")

            # gr.Image for file upload functionality
            # When no image is provided, shows upload interface
            # label: Text shown above upload area
            # height: Consistent sizing with generated images
            uploaded_image = gr.Image(label="Upload Food Image", height=400)

            # Secondary action button (less prominent than primary)
            # variant="secondary": Usually gray/muted color scheme
            analyze_btn = gr.Button("🔍 Analyze Food", variant="secondary")

        # Right column for analysis results
        with gr.Column(scale=1):
            # Output textbox for displaying analysis results
            # Same configuration as generation section for consistency
            uploaded_analysis = gr.Textbox(
                label="Analysis Results",
                lines=10,  # Multi-line for detailed results
                interactive=False,  # Read-only output
            )

    # Event handler for analyze button
    # inputs: Single component (not a list since only one input)
    # outputs: Single component (not a list since only one output)
    analyze_btn.click(
        fn=analyze_uploaded_food,  # Function to execute
        inputs=uploaded_image,  # Image component as input
        outputs=uploaded_analysis,  # Textbox component as output
    )

    # Final section with examples and usage tips
    # Triple-quoted string allows multi-line markdown content
    gr.Markdown(
        """
    ---
    ### 📝 Example Food Descriptions:
    - veg noodles
    - chilli garlic veg noodles
    - chicken noodles
    - schezwan chicken noodles
    - grilled fish with vegetables
    - veg margherita pizza
    - chicken caesar salad
    - veg stir fry
    - chocolate cake with strawberries
    
    
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