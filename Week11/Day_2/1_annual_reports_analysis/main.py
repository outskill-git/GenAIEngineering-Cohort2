"""
Financial Advisor RAG Crew - Uses existing FAISS index only
NO EMBEDDING GENERATION - Will fail if index doesn't exist
NO MODEL LOADING - Uses dummy embeddings for compatibility
Uses YAML configuration files for agents and tasks
"""

from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, tool, crew, before_kickoff, after_kickoff
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import pickle
import hashlib
from pathlib import Path
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from pydantic import BaseModel, Field
import pandas as pd
from dotenv import load_dotenv
import warnings
import json
import faiss
import numpy as np
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LITELLM_LOG'] = 'DEBUG'
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

llm = LLM(
    model='openai/gpt-4o',
    api_key=os.getenv('OPEN_ROUTER_KEY'),
    base_url="https://openrouter.ai/api/v1"
)


# Custom BaseTool implementation to avoid crewai_tools import issues
class BaseTool(ABC):
    """Custom BaseTool implementation to avoid crewai_tools import issues"""
    name: str = ""
    description: str = ""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _run(self, *args, **kwargs):
        """Run the tool"""
        pass

    def run(self, **kwargs):
        """Public run method that CrewAI will call"""
        return self._run(**kwargs)


class VectorSearchTool(BaseTool):
    """Custom tool to search the FAISS vector store"""

    def __init__(self, vector_store):
        super().__init__()
        self.name = "vector_search"
        self.description = "Search through annual reports for specific information using semantic search. Use this to find financial data, metrics, statements, and any information from the reports."
        self.vector_store = vector_store

    def _run(self, query: str, num_results: int = 5, **kwargs) -> str:
        """Execute the search"""
        if not self.vector_store:
            return "Error: Vector store not initialized"

        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=num_results)

            if not results:
                return f"No results found for query: {query}"

            # Format results
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get('source', 'Unknown')
                chunk_index = doc.metadata.get('chunk_index', 'Unknown')

                formatted_results.append(
                    f"Result {i} (Score: {score:.3f}):\n"
                    f"Source: {source}, Chunk: {chunk_index}\n"
                    f"Content: {doc.page_content}\n"
                    f"{'-' * 50}"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            return f"Error searching vector store: {str(e)}"


class CustomFAISSLoader:
    """Adapter to load the notebook-created FAISS index into LangChain format"""

    @staticmethod
    def load_notebook_index(index_path: str, embeddings) -> LangchainFAISS:
        """Load FAISS index created by the notebook and convert to LangChain format"""

        # Load the raw FAISS index
        index = faiss.read_index(os.path.join(index_path, "index.faiss"))

        # Load chunks
        with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)

        # Load metadata
        with open(os.path.join(index_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Create documents for LangChain
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_id": chunk.get("chunk_id", f"{i}")
                }
            )
            documents.append(doc)

        # Create docstore
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})

        # Create index to docstore id mapping
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        # Create LangChain FAISS instance
        vector_store = LangchainFAISS(
            embedding_function=embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        return vector_store


@CrewBase
class FinancialAdvisorRAGCrew():
    """
    An advanced financial advisory crew that analyzes annual reports using RAG,
    compares companies, and provides investment recommendations based on fundamental analysis.

    THIS VERSION ONLY USES EXISTING FAISS INDEX - NO EMBEDDING GENERATION
    NO MODEL LOADING - Uses dummy embeddings for compatibility
    """

    # Point to the config files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        """Initialize tools and vector store for RAG"""
        self.reports_path = "annual_reports"
        self.vector_store_path = "vector_store"
        self.notebook_index_path = "faiss_index"  # Path from your notebook
        self.vector_store = None
        self.vector_search_tool = None
        self.embeddings = None

        # Create necessary directories
        Path(self.vector_store_path).mkdir(exist_ok=True)
        Path(self.reports_path).mkdir(exist_ok=True)

        # Check for existing index FIRST
        print("\n🔍 Checking for existing FAISS index...")
        if not self._check_for_existing_index():
            print("\n❌ FATAL: No FAISS index found!")
            print("   This version requires a pre-built FAISS index.")
            print("\n   Options:")
            print("   1. Run the Jupyter notebook to create an index")
            print("   2. Use a version of this script that creates embeddings")
            print("\n   Expected index location: faiss_index/")
            print("   Required files:")
            print("   - faiss_index/index.faiss")
            print("   - faiss_index/chunks.pkl")
            print("   - faiss_index/metadata.json")
            raise RuntimeError("No FAISS index found. Cannot proceed without embeddings.")

        # Initialize embeddings - Just a compatibility layer, no model loading
        self._initialize_embeddings()

        # Setup RAG system (will only load, not create)
        self._setup_rag_system()

        # Initialize CrewBase parent
        super().__init__()

    def _check_for_existing_index(self) -> bool:
        """Check if a FAISS index exists"""
        # Check for notebook index
        notebook_files = ["index.faiss", "chunks.pkl", "metadata.json"]
        notebook_exists = all(
            os.path.exists(os.path.join(self.notebook_index_path, f))
            for f in notebook_files
        )

        # Check for LangChain index
        langchain_exists = os.path.exists(
            os.path.join(self.vector_store_path, "faiss_index.faiss")
        )

        return notebook_exists or langchain_exists

    def _initialize_embeddings(self):
        """Initialize embeddings - Just create a dummy function for loading, no model download"""
        try:
            print("🔍 Setting up embedding compatibility layer...")

            # Read the embedding dimension from the existing index metadata
            metadata_file = os.path.join(self.notebook_index_path, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                embedding_dim = metadata.get('embedding_dim', 384)
                print(f"   Using embedding dimension from index: {embedding_dim}")
            else:
                embedding_dim = 384  # Default for all-MiniLM-L6-v2
                print(f"   Using default embedding dimension: {embedding_dim}")

            # Create a dummy embedding class that doesn't load any model
            class DummyEmbeddings:
                def __init__(self, dim):
                    self.dim = dim

                def embed_query(self, text: str) -> List[float]:
                    """Dummy embedding function - returns zeros
                    We never actually use this since we only load existing index"""
                    return [0.0] * self.dim

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    """Dummy embedding function for documents"""
                    return [[0.0] * self.dim for _ in texts]

            self.embeddings = DummyEmbeddings(embedding_dim)
            print("✅ Embedding compatibility layer ready (no model loaded)")

        except Exception as e:
            print(f"❌ Error setting up embeddings: {str(e)}")
            raise

    def _setup_rag_system(self):
        """Setup the RAG system with vector store and tools - LOAD ONLY, NO CREATION"""
        print("\n📂 Loading existing FAISS index...")

        # Try to load existing vector store
        if not self._load_vector_store():
            raise RuntimeError("Failed to load FAISS index. Cannot proceed.")

        print("✅ Successfully loaded FAISS index!")
        print("🔧 VectorSearchTool ready for agents to use")

    def _load_vector_store(self) -> bool:
        """Load vector store from disk - NO CREATION, ONLY LOADING"""

        # First, try to load the notebook-created index
        if os.path.exists(os.path.join(self.notebook_index_path, "index.faiss")):
            try:
                print(f"📂 Found notebook FAISS index at '{self.notebook_index_path}/'")

                # Load metadata to show info
                metadata_file = os.path.join(self.notebook_index_path, "metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"   Total chunks: {metadata.get('total_chunks', 'Unknown')}")
                    print(f"   Documents: {', '.join(metadata.get('documents', []))}")

                print("   Loading index...")

                self.vector_store = CustomFAISSLoader.load_notebook_index(
                    self.notebook_index_path,
                    self.embeddings
                )

                # Create vector search tool
                self.vector_search_tool = VectorSearchTool(vector_store=self.vector_store)

                print("✅ Successfully loaded notebook FAISS index")
                return True

            except Exception as e:
                print(f"❌ Error loading notebook index: {str(e)}")
                return False

        # Try LangChain format
        store_file = os.path.join(self.vector_store_path, "faiss_index.faiss")
        if os.path.exists(store_file):
            try:
                print(f"📂 Found LangChain FAISS index at '{self.vector_store_path}/'")
                print("   Loading index...")

                # Load the vector store
                self.vector_store = LangchainFAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    "faiss_index"
                )

                # Create vector search tool
                self.vector_search_tool = VectorSearchTool(vector_store=self.vector_store)

                print("✅ Successfully loaded LangChain vector store")
                return True

            except Exception as e:
                print(f"❌ Error loading LangChain index: {str(e)}")
                return False

        print("❌ No FAISS index found in any expected location")
        return False

    @before_kickoff
    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and validate inputs before the crew starts"""
        inputs['current_date'] = datetime.now().strftime("%Y-%m-%d")

        # Add available reports to context
        if os.path.exists(self.reports_path):
            reports = [f for f in os.listdir(self.reports_path) if f.endswith('.pdf')]
            inputs['available_reports'] = reports
            inputs['reports_count'] = len(reports)
        else:
            inputs['available_reports'] = []
            inputs['reports_count'] = 0

        # Set default analysis parameters if not provided
        if 'analysis_focus' not in inputs:
            inputs['analysis_focus'] = [
                'revenue growth',
                'profitability',
                'debt levels',
                'competitive positioning',
                'future outlook'
            ]

        if 'investment_criteria' not in inputs:
            inputs['investment_criteria'] = {
                'min_revenue_growth': 10,  # %
                'min_profit_margin': 15,   # %
                'max_debt_to_equity': 1.5,
                'investment_horizon': '3-5 years',
                'risk_tolerance': 'moderate'
            }

        # Ensure risk_tolerance is always present at the top level
        if 'risk_tolerance' not in inputs:
            if 'investment_criteria' in inputs and 'risk_tolerance' in inputs['investment_criteria']:
                inputs['risk_tolerance'] = inputs['investment_criteria']['risk_tolerance']
            else:
                inputs['risk_tolerance'] = 'moderate'

        # Add all investment criteria fields to top level for template access
        if 'investment_criteria' in inputs:
            for key, value in inputs['investment_criteria'].items():
                if key not in inputs:
                    inputs[key] = value

        # Ensure comparison_metrics exists
        if 'comparison_metrics' not in inputs:
            inputs['comparison_metrics'] = [
                'revenue_growth_rate',
                'net_profit_margin',
                'debt_to_equity_ratio',
                'return_on_equity'
            ]

        return inputs

    @after_kickoff
    def process_output(self, output):
        """Process and format the output after crew completion"""
        disclaimer = "\n\n---\nDISCLAIMER: This analysis is based on historical data from annual reports and should not be considered as personalized investment advice. Past performance does not guarantee future results. Please consult with a qualified financial advisor before making investment decisions."
        output.raw += f"\n\nAnalysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        output.raw += disclaimer
        return output



    @agent
    def document_analyst(self) -> Agent:
        """Document Analyst Agent - Extracts key information from annual reports"""
        print(self.vector_search_tool)
        return Agent(
            config=self.agents_config['document_analyst'],
            verbose=True,
            llm=llm  # Use our OpenRouter LLM
        )

    @agent
    def financial_analyst(self) -> Agent:
        """Financial Analyst Agent - Analyzes financial metrics and ratios"""
        return Agent(
            config=self.agents_config['financial_analyst'],
            verbose=True,
            llm=llm  # Use our OpenRouter LLM
        )

    @agent
    def comparative_analyst(self) -> Agent:
        """Comparative Analyst Agent - Compares multiple companies"""
        return Agent(
            config=self.agents_config['comparative_analyst'],
            verbose=True,
            llm=llm  # Use our OpenRouter LLM
        )

    @agent
    def investment_strategist(self) -> Agent:
        """Investment Strategist Agent - Provides investment recommendations"""
        return Agent(
            config=self.agents_config['investment_strategist'],
            tools=[],  # This agent typically doesn't need search tools
            verbose=True,
            llm=llm  # Use our OpenRouter LLM
        )

    @agent
    def risk_analyst(self) -> Agent:
        """Risk Analyst Agent - Identifies and evaluates investment risks"""
        return Agent(
            config=self.agents_config['risk_analyst'],
            verbose=True,
            llm=llm  # Use our OpenRouter LLM
        )

    @task
    def extract_report_data(self) -> Task:
        """Task to extract key data from annual reports"""
        return Task(
            config=self.tasks_config['extract_report_data']
        )

    @task
    def analyze_financials(self) -> Task:
        """Task to analyze financial metrics and calculate ratios"""
        return Task(
            config=self.tasks_config['analyze_financials']
        )

    @task
    def compare_companies(self) -> Task:
        """Task to compare companies based on extracted data"""
        return Task(
            config=self.tasks_config['compare_companies']
        )

    @task
    def assess_risks(self) -> Task:
        """Task to identify and assess investment risks"""
        return Task(
            config=self.tasks_config['assess_risks']
        )

    @task
    def generate_recommendations(self) -> Task:
        """Task to generate investment recommendations"""
        return Task(
            config=self.tasks_config['generate_recommendations']
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the financial advisor RAG crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def search_reports(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the vector store for relevant information"""
        if self.vector_store:
            results = self.vector_store.similarity_search_with_score(query, k=k)

            # Format results with metadata
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'chunk_index': doc.metadata.get('chunk_index', -1),
                    'score': float(score),
                    'preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })

            return formatted_results
        return []

    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the current vector store"""
        info = {
            'initialized': self.vector_store is not None,
            'vector_search_tool_ready': self.vector_search_tool is not None,
            'embedding_type': 'Index-only mode (no model loaded)',
            'reports_path': self.reports_path,
            'vector_store_path': self.vector_store_path,
            'pdf_count': 0,
            'reports': []
        }

        # Count PDFs
        if os.path.exists(self.reports_path):
            pdf_files = [f for f in os.listdir(self.reports_path) if f.endswith('.pdf')]
            info['pdf_count'] = len(pdf_files)
            info['reports'] = pdf_files

        # Check notebook index
        notebook_index = os.path.join(self.notebook_index_path, "index.faiss")
        if os.path.exists(notebook_index):
            info['notebook_index_found'] = True
            info['notebook_index_size_mb'] = round(os.path.getsize(notebook_index) / (1024 * 1024), 2)

            # Load metadata if available
            metadata_file = os.path.join(self.notebook_index_path, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    nb_metadata = json.load(f)
                info['notebook_chunks'] = nb_metadata.get('total_chunks', 'Unknown')
                info['notebook_documents'] = nb_metadata.get('documents', [])

        return info

    def test_vector_search_tool(self):
        """Test the vector search tool directly"""
        if not self.vector_search_tool:
            print("❌ Vector search tool not initialized")
            return

        print("\n🔧 Testing VectorSearchTool...")
        test_query = "revenue growth and profitability"
        result = self.vector_search_tool._run(query=test_query, num_results=3)
        print(f"Query: '{test_query}'")
        print("Results:")
        print(result)


# Example usage
if __name__ == "__main__":
    # Check for OpenRouter API key (required for agents)
    if not os.getenv('OPEN_ROUTER_KEY'):
        print("❌ Error: OPEN_ROUTER_KEY not set.")
        print("   Set it using: export OPEN_ROUTER_KEY='your-key-here'")
        print("   Or add it to your .env file")
        exit(1)

    print("="*60)
    print("🚀 Financial Advisor RAG Crew")
    print("   NO EMBEDDING GENERATION VERSION")
    print("   NO MODEL LOADING - Using existing index only")
    print("   Using YAML configuration files")
    print("="*60)

    # Check for YAML files
    if not os.path.exists('config/agents.yaml'):
        print("❌ Error: config/agents.yaml not found!")
        print("   Please ensure your config files are in the 'config' directory")
        exit(1)

    if not os.path.exists('config/tasks.yaml'):
        print("❌ Error: config/tasks.yaml not found!")
        print("   Please ensure your config files are in the 'config' directory")
        exit(1)

    print("✅ Configuration files found")

    try:
        # Initialize the crew - will fail if no index exists
        financial_crew = FinancialAdvisorRAGCrew()

        # Display vector store info
        info = financial_crew.get_vector_store_info()
        print("\n📊 Vector Store Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")

        # Test vector search tool
        print("\n🔍 Testing vector search functionality...")
        financial_crew.test_vector_search_tool()

        # Analysis parameters
        analysis_inputs = {
            'analysis_focus': [
                'revenue growth trends',
                'profitability metrics',
                'cash flow analysis',
                'debt and liquidity',
                'competitive advantages',
                'growth prospects',
                'management effectiveness',
                'market positioning'
            ],
            'investment_criteria': {
                'min_revenue_growth': 15,      # %
                'min_profit_margin': 20,       # %
                'max_debt_to_equity': 1.0,
                'min_roe': 15,                 # % Return on Equity
                'investment_horizon': '5 years',
                'risk_tolerance': 'moderate'
            },
            'comparison_metrics': [
                'revenue_growth_rate',
                'ebitda_margin',
                'net_profit_margin',
                'debt_to_equity_ratio',
                'return_on_equity',
                'return_on_assets',
                'current_ratio',
                'free_cash_flow'
            ],
            'sectors_of_interest': [
                'technology',
                'healthcare',
                'consumer goods'
            ]
        }

        # Run the crew
        run_analysis = input("\n🎯 Run full analysis? (y/n): ").lower().strip() == 'y'
        if run_analysis:
            print("\n🎯 Starting analysis...")
            print("   Agents will use VectorSearchTool to find information in the reports")
            result = financial_crew.crew().kickoff(inputs=analysis_inputs)

            # Print the result
            print("\n" + "="*60)
            print("ANNUAL REPORT ANALYSIS & INVESTMENT RECOMMENDATIONS")
            print("="*60)
            print(result)

            # Save result to file
            output_file = f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(output_file, 'w') as f:
                f.write(str(result))
            print(f"\n📄 Full analysis saved to: {output_file}")

    except RuntimeError as e:
        print(f"\n❌ {str(e)}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)