# Week 10: Vector Databases and RAG Pipelines with LanceDB and Gradio

This week focuses on understanding and implementing vector databases using LanceDB and building complete RAG (Retrieval-Augmented Generation) pipelines for multimodal applications. You'll learn to create sophisticated search engines that combine vector similarity search with large language models.

## Learning Objectives

- Understand vector databases and their role in AI applications
- Learn to use LanceDB for efficient vector storage and retrieval
- Build multimodal search engines with text and image capabilities
- Implement complete RAG pipelines with retrieval, augmentation, and generation
- Create interactive web applications using Gradio
- Work with fashion/e-commerce datasets for practical applications
- Integrate multiple AI models (CLIP, LLMs) in production-ready systems
- Understand prompt engineering and context augmentation techniques

## Repository Setup

### Clone the GitHub Repository

If you haven't already, clone the repository using Git:

```bash
# Clone the repository
git clone https://github.com/outskill-git/GenAIEngineering-Cohort1

# Navigate into the repository folder
cd GenAIEngineering-Cohort1/Week9
```

### Create a Virtual Environment

Create and activate a Python virtual environment:

#### For Windows:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

#### For macOS/Linux:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Install Requirements

Install the packages listed in requirements.txt:

```bash
# Install required packages
pip install -r requirements.txt
```

### Verify Installation

Verify that everything is set up correctly:

```bash
# Check installed packages
pip list

# Test key imports
python -c "import lancedb; import torch; import transformers; import gradio; print('Setup successful!')"
```

## Session Content

### Session 1: LanceDB Multimodal Fashion Search Engine

**Gradio Introduction Notebook"** `introduction_to_gradio.ipynb` [Colab](https://colab.research.google.com/drive/13d7MRgylFJZ9NCrfLeK0_mD1nh_Qx3d7?usp=sharing)

**Main Notebook:** `lancedb_multimodal_myntra_fashion_search_engine.ipynb` [Colab](https://colab.research.google.com/drive/17CNo2rkbFYaIYcS5_ABd-fimBDWBi0C7?usp=sharing)

#### Data Requirements for lancedb_multimodal_myntra_fashion_search_engine.ipynb:
- For this project you need to download the [Myntra Fashion Product Dataset]( https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset) from Kaggle.
- Create a folder named `input` within `session_1`.
- Unzip the downloaded data and move it in the `input` folder.
- The final directory structure should look like this
```python
Week9
  |-session_1
  |     |-input
  |         |-Fashion Dataset.csv
  |         |-Images
  |             |-Images
  |                |-0.jpg
  |                |-2.jpg
  |                .
  |                .
  |                .
  |-session_2
        |-....
```

#### Key Components:

1. **Vector Database Setup with LanceDB**
   - Creating and managing vector databases
   - Schema definition using Pydantic models
   - Embedding registration and model integration
   - Table creation and data ingestion

2. **Multimodal Embeddings with CLIP**
   - Using OpenCLIP for text and image embeddings
   - Creating unified vector representations
   - Similarity search across modalities

3. **Fashion Dataset Integration**
   - Working with Myntra Fashion Product Dataset
   - Image preprocessing and metadata handling
   - Building searchable product catalogs

4. **Applications Built:**
   - **food_app.py**: [Colab](https://colab.research.google.com/drive/1GKJUa7zI9Ei0IkgNzRlBEhijdILhdymc?usp=sharing) [HuggingFace Spaces](https://huggingface.co/spaces/ishandutta/multimodal-food-image-generation-and-analysis) Food product generation and analysis
   - **product_cataloger_app.py**: [Colab](https://colab.research.google.com/drive/1eFNaidx5TPEhXgzdY9hh7EhDcVZm4GMS?usp=sharing) [HuggingFace Spaces](https://huggingface.co/spaces/ishandutta/multimodal-product-cataloger) Advanced product cataloging

### Session 2: Complete RAG Pipeline Implementation

**Main Files:**
- `app.py`: Gradio web interface for RAG pipeline
- `rag_pipeline.py`: Complete RAG pipeline integration
- `retriever.py`: Vector search and data management
- `generator.py`: LLM setup and response generation  
- `augmenter.py`: Context enhancement and prompt engineering
  
[Colab Notebook](https://colab.research.google.com/drive/1rq-ywjykHBw7xPXCmd3DmZdK6T9bhDtA?usp=sharing)
  
[HuggingFace Spaces](https://huggingface.co/spaces/ishandutta/multimodal-myntra-shoes-rag-pipeline)
  
#### Key Components:

1. **Retrieval System (`retriever.py`)**
   - Vector database management with LanceDB
   - Multimodal search (text and image queries)
   - Efficient similarity search and ranking
   - Schema design for shoe/fashion products

2. **Augmentation Engine (`augmenter.py`)**
   - Query classification and analysis
   - Context enhancement with retrieved results
   - Advanced prompt engineering techniques
   - Dynamic prompt generation based on query type

3. **Generation Module (`generator.py`)**
   - Integration with multiple LLM providers (Qwen, OpenAI)
   - Response generation with retrieved context
   - Model management and optimization
   - Flexible model switching capabilities

4. **Complete RAG Pipeline (`rag_pipeline.py`)**
   - End-to-end pipeline integration
   - Retrieval → Augmentation → Generation workflow
   - Detailed step tracking and logging
   - Error handling and fallback mechanisms

5. **Web Interface (`app.py`)**
   - Interactive Gradio web application
   - Multimodal input support (text/image)
   - Real-time pipeline visualization
   - Step-by-step execution tracking

#### Data Requirements:
- Hugging Face shoe dataset (automatically downloaded) 

## Key Concepts Covered

### Vector Databases
- **LanceDB**: High-performance vector database for AI applications
- **Vector Storage**: Efficient storage and indexing of high-dimensional embeddings
- **Similarity Search**: Fast approximate nearest neighbor search
- **Schema Design**: Pydantic-based schema definition for structured data
- **Scalability**: Handling large-scale vector datasets

### RAG (Retrieval-Augmented Generation)
- **Retrieval Phase**: Vector similarity search and result ranking
- **Augmentation Phase**: Context enhancement and prompt engineering
- **Generation Phase**: LLM-based response generation
- **Pipeline Integration**: Seamless component integration
- **Performance Optimization**: Efficient pipeline execution

### Multimodal AI
- **CLIP Integration**: Unified text-image understanding
- **Cross-modal Search**: Finding images from text and vice versa
- **Embedding Alignment**: Shared vector space for different modalities
- **Multimodal Applications**: Real-world use cases in e-commerce

### Prompt Engineering
- **Query Classification**: Automatic query type detection
- **Context Formatting**: Structured information presentation
- **Dynamic Prompts**: Adaptive prompt generation
- **Advanced Techniques**: Context-aware prompt optimization

### Web Applications
- **Gradio Framework**: Rapid web interface development
- **Interactive Components**: Real-time user interaction
- **Visualization**: Step-by-step pipeline visualization
- **User Experience**: Intuitive multimodal interfaces

### LLM Integration
- **Multiple Providers**: Support for Qwen, OpenAI, and other models
- **Model Management**: Efficient model loading and switching
- **Response Generation**: Context-aware text generation
- **Error Handling**: Robust fallback mechanisms

## Running the Applications

### Session 1 Applications:

```bash
# Run food recommendation app
cd session_1
python food_app.py

# Run product cataloger app
python product_cataloger_app.py
```

### Session 2 RAG Pipeline:

```bash
cd session_2

# Setup database and run complete pipeline
python rag_pipeline.py --setup-db --query "recommend me men's running shoes"

# Run with image query
python rag_pipeline.py --query "hf_shoe_images/shoe_0000.jpg"

# Run with OpenAI (requires API key)
python rag_pipeline.py --query "comfortable sneakers" --model-provider openai --openai-api-key YOUR_KEY

# Launch interactive web interface
python app.py

# Launch with custom settings
python app.py --port 7860 --share
```

### Command Line Options:

```bash
# RAG Pipeline Options
--query: Search query (text or image path)
--model-provider: Choose between 'qwen' or 'openai' 
--model-name: Specific model to use
--openai-api-key: API key for OpenAI models
--setup-db: Initialize database before running
--no-llm: Retrieval only (no generation)
--detailed-steps: Enable detailed step tracking

# Web App Options
--port: Custom port number
--host: Custom host address
--share: Enable public sharing
```

## Technical Architecture

### Database Layer
- **LanceDB**: Vector storage and retrieval
- **Schema Management**: Pydantic model definitions
- **Indexing**: Optimized vector indexing for fast search
- **Data Ingestion**: Batch processing of large datasets

### AI/ML Layer
- **Embedding Models**: CLIP for multimodal embeddings
- **Language Models**: Qwen, OpenAI for text generation
- **Vector Search**: Cosine similarity and ranking
- **Model Management**: Efficient model loading and caching

### Application Layer
- **Pipeline Orchestration**: RAG workflow management
- **Web Interface**: Gradio-based user interface
- **API Integration**: External model API handling
- **Error Management**: Comprehensive error handling

### Data Flow
1. **Input Processing**: Query/image preprocessing
2. **Embedding Generation**: Convert inputs to vectors
3. **Vector Search**: Find similar items in database
4. **Context Augmentation**: Enhance retrieved results
5. **Response Generation**: LLM-based final response
6. **Output Formatting**: User-friendly result presentation

## Troubleshooting

### Common Issues:

1. **Database Connection Errors**
   ```bash
   # Clear database cache
   rm -rf .lancedb/
   python rag_pipeline.py --setup-db
   ```

2. **Model Loading Issues**
   ```bash
   # Check available models
   python -c "from generator import get_available_models; print(get_available_models())"
   ```

3. **Memory Issues**
   ```bash
   # Use smaller batch sizes or lighter models
   # Reduce sample_size in create_table function
   ```
   
### Getting Help:

- Check that your virtual environment is activated
- Verify all packages are installed with `pip list`
- Ensure sufficient disk space for model downloads and databases
- Check model compatibility with your hardware
- Monitor memory usage for large datasets

## Additional Resources

- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [OpenCLIP Documentation](https://github.com/mlfoundations/open_clip) 