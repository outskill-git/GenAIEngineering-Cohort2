import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.config import RunnableConfig

import json
import asyncio
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import re
from datetime import datetime
import logging
import requests

# Langsmith integration - initialized at runtime
import langsmith
from dotenv import load_dotenv

# Global variables for Langsmith
_langsmith_initialized = False
_langsmith_client = None

def _initialize_langsmith():
    """Initialize Langsmith tracing - called at runtime"""
    global _langsmith_initialized, _langsmith_client

    if _langsmith_initialized:
        return _langsmith_client

    try:
        # Load environment variables
        load_dotenv()

        # Check API keys
        langsmith_key = os.getenv('LANGSMITH_API_KEY')
        if not langsmith_key:
            logger.warning("LANGSMITH_API_KEY not found in environment")
            return None

        # Create Langsmith client with explicit API key
        _langsmith_client = langsmith.Client(api_key=langsmith_key)

        # Set environment variables for tracing (CORRECT format from docs)
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = langsmith_key
        os.environ["LANGSMITH_PROJECT"] = "code-review-app"
        # Set for backward compatibility too
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "code-review-app"

        _langsmith_initialized = True
        logger.info("✅ Langsmith tracing initialized successfully")
        return _langsmith_client

    except Exception as e:
        logger.error(f"Failed to initialize Langsmith: {e}")
        return None

def get_langsmith_client():
    """Get Langsmith client (initialize if needed)"""
    return _initialize_langsmith()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for structured output
class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class IssueCategory(str, Enum):
    SECURITY = "Security"
    PERFORMANCE = "Performance"
    BUG = "Bug"
    STYLE = "Style"
    MAINTAINABILITY = "Maintainability"

class CodeIssue(BaseModel):
    """Structured code issue model for LangChain output parsing"""
    severity: Severity
    category: IssueCategory
    line_number: int = Field(description="Line number where issue occurs")
    title: str = Field(description="Brief issue title")
    description: str = Field(description="Detailed issue description")
    suggestion: str = Field(description="How to fix the issue")
    code_snippet: Optional[str] = Field(default="", description="Relevant code snippet")
    confidence: float = Field(ge=0.0, le=1.0, description="AI confidence in this issue")

class AnalysisResult(BaseModel):
    """Complete analysis result structure"""
    issues: List[CodeIssue]
    summary: str
    quality_score: float = Field(ge=0.0, le=100.0)
    model_used: str
    analysis_time: float
    recommendations: List[str]
    token_usage: Dict[str, Union[int, float]] = Field(default_factory=dict)

# OpenRouter model configurations
OPENROUTER_MODELS = {
    # Free Models
    "Grok 4 Fast (Free)": {
        "id": "x-ai/grok-4-fast:free",
        "context": "32K tokens",
        "cost": "FREE - No credits required",
        "provider": "xAI"
    },
    "Grok 4o Mini (Free)": {
        "id": "x-ai/grok-4o-mini:free",
        "context": "32K tokens",
        "cost": "FREE - Fast and efficient",
        "provider": "xAI"
    },

    # Anthropic Models
    "Claude 3.5 Sonnet": {
        "id": "anthropic/claude-3.5-sonnet",
        "context": "200K tokens",
        "cost": "High quality, excellent for code analysis",
        "provider": "Anthropic"
    },
    "Claude 3 Opus": {
        "id": "anthropic/claude-3-opus",
        "context": "200K tokens", 
        "cost": "Premium, best reasoning",
        "provider": "Anthropic"
    },
    "Claude 3 Haiku": {
        "id": "anthropic/claude-3-haiku",
        "context": "200K tokens",
        "cost": "Fast and economical",
        "provider": "Anthropic"
    },
    "Claude 3.5 Haiku": {
        "id": "anthropic/claude-3-5-haiku",
        "context": "200K tokens",
        "cost": "Fast, latest Haiku model",
        "provider": "Anthropic"
    },
    
    # OpenAI Models
    "GPT-4o": {
        "id": "openai/gpt-4o",
        "context": "128K tokens",
        "cost": "Latest GPT-4, excellent performance",
        "provider": "OpenAI"
    },
    "GPT-4o Mini": {
        "id": "openai/gpt-4o-mini",
        "context": "128K tokens",
        "cost": "Fast and cost-effective GPT-4",
        "provider": "OpenAI"
    },
    "GPT-4 Turbo": {
        "id": "openai/gpt-4-turbo",
        "context": "128K tokens",
        "cost": "High quality, good for complex analysis",
        "provider": "OpenAI"
    },
    "GPT-4": {
        "id": "openai/gpt-4",
        "context": "8K tokens",
        "cost": "Premium quality, shorter context",
        "provider": "OpenAI"
    },
    "GPT-3.5 Turbo": {
        "id": "openai/gpt-3.5-turbo",
        "context": "16K tokens",
        "cost": "Economical, good for basic analysis",
        "provider": "OpenAI"
    },
    
    # Google Models
    "Gemini Pro 1.5": {
        "id": "google/gemini-pro-1.5",
        "context": "2M tokens",
        "cost": "Latest Gemini, massive context",
        "provider": "Google"
    },
    "Gemini Pro": {
        "id": "google/gemini-pro",
        "context": "128K tokens",
        "cost": "Good balance of quality and cost",
        "provider": "Google"
    },
    "Gemini Flash": {
        "id": "google/gemini-flash-1.5",
        "context": "1M tokens",
        "cost": "Fast Gemini with large context",
        "provider": "Google"
    },
    "Gemini 2.0 Flash": {
        "id": "google/gemini-2.0-flash-exp",
        "context": "1M tokens",
        "cost": "Experimental, very large context",
        "provider": "Google"
    },
    
    # Meta Models
    "Llama 3.1 70B": {
        "id": "meta-llama/llama-3.1-70b-instruct",
        "context": "128K tokens",
        "cost": "Open source, good performance",
        "provider": "Meta"
    },
    "Llama 3.1 8B": {
        "id": "meta-llama/llama-3.1-8b-instruct",
        "context": "128K tokens",
        "cost": "Fast and lightweight",
        "provider": "Meta"
    },
    "Llama 3.1 405B": {
        "id": "meta-llama/llama-3.1-405b-instruct",
        "context": "128K tokens",
        "cost": "Largest open model, premium performance",
        "provider": "Meta"
    },

    # Mistral Models
    "Mistral 7B": {
        "id": "mistralai/mistral-7b-instruct",
        "context": "32K tokens",
        "cost": "Fast and capable",
        "provider": "Mistral"
    },
    "Mixtral 8x7B": {
        "id": "mistralai/mixtral-8x7b-instruct",
        "context": "32K tokens",
        "cost": "Mixture of experts, high quality",
        "provider": "Mistral"
    },
    "Codestral": {
        "id": "mistralai/codestral-mamba",
        "context": "256K tokens",
        "cost": "Code-focused model",
        "provider": "Mistral"
    },
    
    # Specialized Models
    "DeepSeek Coder": {
        "id": "deepseek/deepseek-coder",
        "context": "16K tokens",  
        "cost": "Specialized for code analysis",
        "provider": "DeepSeek"
    },
    "DeepSeek V2.5": {
        "id": "deepseek/deepseek-chat",
        "context": "32K tokens",
        "cost": "Advanced reasoning capabilities",
        "provider": "DeepSeek"
    },

    # Other Models
    "Qwen 2.5 72B": {
        "id": "qwen/qwen-2.5-72b-instruct",
        "context": "32K tokens",
        "cost": "High-performance open model",
        "provider": "Qwen"
    },
    "Gemma 7B": {
        "id": "google/gemma-7b-it",
        "context": "8K tokens",
        "cost": "Google's lightweight model",
        "provider": "Google"
    }
}

class OpenRouterCallback(BaseCallbackHandler):
    """Custom callback for tracking OpenRouter usage"""
    def __init__(self):
        self.model_calls = []
        self.total_tokens = 0
        self.total_cost = 0.0
        self.current_model = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self.model_calls.append({
            'timestamp': datetime.now(),
            'model': serialized.get('model_name', 'unknown'),
            'prompt_length': sum(len(p) for p in prompts)
        })
        self.current_model = serialized.get('model_name', 'unknown')
    
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            tokens = usage.get('total_tokens', 0)
            self.total_tokens += tokens
            
            # Estimate cost (rough estimates)
            cost_estimates = {
                'gpt-4': 0.03 * tokens / 1000,
                'gpt-3.5-turbo': 0.002 * tokens / 1000,
                'claude-3': 0.015 * tokens / 1000,
                'gemini': 0.001 * tokens / 1000
            }
            
            model_family = self.current_model.split('/')[0] if '/' in self.current_model else self.current_model
            estimated_cost = cost_estimates.get(model_family, 0.01 * tokens / 1000)
            self.total_cost += estimated_cost

class OpenRouterLangChainReviewer:
    """LangChain-based code reviewer using OpenRouter API"""
    
    def __init__(self):
        self.llm = None
        self.current_model = None
        self.api_key = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "class ", "def ", "function ", "//", "#", "```"]
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.callback = OpenRouterCallback()
        self._setup_parsers()
        self._setup_prompts()
    
    def _setup_parsers(self):
        """Initialize LangChain output parsers"""
        self.json_parser = JsonOutputParser()
        self.pydantic_parser = PydanticOutputParser(pydantic_object=AnalysisResult)
        self.issue_parser = PydanticOutputParser(pydantic_object=CodeIssue)
        self.str_parser = StrOutputParser()
    
    def _setup_prompts(self):
        """Setup LangChain prompt templates"""
        
        # System message for all analyses
        self.system_message = SystemMessage(content="""
        You are an expert senior software engineer and security auditor with 15+ years of experience.
        You specialize in code review, security analysis, and performance optimization across multiple languages.
        
        Your analysis should be:
        - Thorough but practical and actionable
        - Security-focused when vulnerabilities are present
        - Performance-aware for optimization opportunities
        - Maintainability-oriented for long-term code health
        
        Always provide specific, actionable suggestions with concrete examples when possible.
        Focus on high-impact issues that improve code quality, security, and performance.
        """)
        
        # Analysis prompts are now created dynamically in each method to avoid template issues
    
    def setup_model(self, model_name: str, api_key: str, temperature: float = 0.1) -> bool:
        """Setup OpenRouter model with LangChain"""
        try:
            self.api_key = api_key
            model_id = OPENROUTER_MODELS[model_name]["id"]
            
            # Initialize Langsmith tracing before creating LLM
            langsmith_client = get_langsmith_client()
            if langsmith_client:
                logger.info("✅ Langsmith tracing active")
            else:
                logger.warning("⚠️ Langsmith tracing not available")
            
            # Create ChatOpenAI instance configured for OpenRouter
            self.llm = ChatOpenAI(
                model=model_id,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                max_tokens=4000,
                model_kwargs={
                    "extra_headers": {
                        "HTTP-Referer": "https://streamlit-langchain-code-review.com",
                        "X-Title": "LangChain Code Review Agent"
                    }
                },
                callbacks=[self.callback]
            )
            
            # Test the model
            test_chain = self.llm | self.str_parser
            test_response = test_chain.invoke("Respond with 'Model ready' if you can analyze code.")
            
            if len(test_response) > 5:
                self.current_model = model_name
                logger.info(f"Successfully configured {model_name} with Langsmith tracing")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to setup {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get information about a specific model"""
        return OPENROUTER_MODELS.get(model_name, {})
    
    def analyze_code_comprehensive(self, code: str, language: str) -> AnalysisResult:
        """Comprehensive code analysis using LangChain chains"""
        if not self.llm:
            raise ValueError("No model configured")
        
        # Ensure Langsmith is initialized
        langsmith_client = get_langsmith_client()
        
        start_time = datetime.now()
        
        try:
            # Check if code needs chunking
            if len(code) > 10000:
                return self._analyze_large_code(code, language)
            
            # Run analysis with direct LLM call to avoid template issues
            try:
                # Debug: Log what we're sending to the model
                logger.info(f"Sending to model - Language: {language}, Code length: {len(code)} chars")
                logger.info(f"Code preview: {code[:200]}...")

                # Create the analysis prompt dynamically to ensure proper variable substitution
                analysis_prompt_text = f"""
                You are a code security auditor. Analyze this {language} code VERY CAREFULLY for ANY issues.

                CODE TO ANALYZE:
                ```{language}
                {code}
                ```

                You MUST return ONLY a valid JSON array. Find issues in this code!

                REQUIRED JSON FORMAT:
                [
                    {{
                        "severity": "Critical",
                        "category": "Security",
                        "line_number": 1,
                        "title": "Specific security vulnerability",
                        "description": "Detailed explanation of the security risk and impact",
                        "suggestion": "Concrete steps to fix the vulnerability",
                        "confidence": 0.95
                    }}
                ]

                If no issues found, return: []

                CRITICAL: Look for these specific issues:
                - Hardcoded passwords/API keys (security vulnerability)
                - Debug mode enabled in production (security risk)
                - CORS allowing all origins (*) (security vulnerability)
                - No authentication on sensitive endpoints (security vulnerability)
                - Logging sensitive data (security vulnerability)
                - SQL injection vulnerabilities
                - Command injection risks
                - Insecure default configurations

                BE SPECIFIC: Reference actual line numbers and quote problematic code.
                NEVER return empty array unless code is truly perfect.
                Your response must be ONLY the JSON array, nothing else.
                """

                # Use the LLM directly with the constructed prompt
                messages = [
                    self.system_message,
                    HumanMessage(content=analysis_prompt_text)
                ]

                raw_result = self.llm.invoke(messages)

                # Extract the content from the response
                if hasattr(raw_result, 'content'):
                    raw_result = raw_result.content
                elif isinstance(raw_result, str):
                    pass  # already a string
                else:
                    raw_result = str(raw_result)

                # Enhanced JSON parsing with multiple fallback strategies
                logger.info(f"Raw model response (first 500 chars): {raw_result[:500]}")
                result = self._parse_analysis_response(raw_result, code)
                logger.info(f"Parsed result: {result}")
                        
            except Exception as e:
                logger.warning(f"Chain execution failed: {e}")
                result = []
            
            # Parse issues with error handling
            issues = []
            if isinstance(result, list):
                for issue_data in result:
                    try:
                        # Add code snippet if line number is valid
                        if isinstance(issue_data, dict):
                            line_num = issue_data.get('line_number', 0)
                            if line_num > 0:
                                code_lines = code.split('\n')
                                if line_num <= len(code_lines):
                                    issue_data['code_snippet'] = code_lines[line_num - 1].strip()
                        
                        issue = CodeIssue(**issue_data)
                        issues.append(issue)
                    except Exception as e:
                        logger.warning(f"Failed to parse issue: {e}")
                        continue
            elif isinstance(result, str):
                # Handle case where result is still a string
                issues = self._parse_text_to_issues(result, code)

            # If no issues found and we have code, try to detect obvious issues
            if not issues and code.strip():
                logger.warning("No issues found by AI, checking for obvious security issues")
                issues = self._detect_obvious_issues(code)
            
            # Generate summary using another chain
            summary = self._generate_summary(issues, code, language)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(issues, len(code.split('\n')))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                issues=issues,
                summary=summary,
                quality_score=quality_score,
                model_used=f"{self.current_model} (via OpenRouter)",
                analysis_time=analysis_time,
                recommendations=recommendations,
                token_usage={
                    "total_tokens": 0,  # Langsmith tracks this automatically
                    "estimated_cost": 0.0  # Langsmith tracks this automatically
                }
            )
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    def analyze_security_focused(self, code: str, language: str) -> List[CodeIssue]:
        """Security-focused analysis using specialized chain"""
        if not self.llm:
            raise ValueError("No model configured")
        
        # Ensure Langsmith is initialized
        langsmith_client = get_langsmith_client()
        
        try:
            # Create the security analysis prompt dynamically
            security_prompt_text = f"""
            You are a cybersecurity expert. Perform DEEP security analysis on this {language} code. Return ONLY valid JSON array.

            CODE TO ANALYZE:
            ```{language}
            {code}
            ```

            You MUST return ONLY a valid JSON array. Find security vulnerabilities!

            CRITICAL SECURITY CHECKS:
            - **Injection Attacks**: SQL injection, command injection, code injection
            - **Authentication**: Weak passwords, missing auth, broken authentication
            - **Authorization**: Privilege escalation, access control bypass
            - **Input Validation**: XSS, CSRF, path traversal, buffer overflow
            - **Cryptography**: Weak encryption, exposed secrets, insecure random
            - **Data Exposure**: Information leakage, sensitive data logging
            - **Configuration**: Debug mode, insecure defaults, exposed internals
            - **Network Security**: CORS misconfiguration, host binding issues

            REQUIRED JSON FORMAT:
            [
                {{
                    "severity": "Critical",
                    "category": "Security",
                    "line_number": 1,
                    "title": "Specific security vulnerability",
                    "description": "Detailed explanation of the security risk and impact",
                    "suggestion": "Concrete steps to fix the vulnerability",
                    "confidence": 0.95
                }}
            ]

            BE SPECIFIC: Include exact line numbers and quote problematic code.
            PRIORITIZE: Focus on high-impact security vulnerabilities.
            NEVER return empty array unless code has NO security issues at all.
            Your response must be ONLY the JSON array, nothing else.
            """

            # Use the LLM directly with the constructed prompt
            messages = [
                self.system_message,
                HumanMessage(content=security_prompt_text)
            ]

            raw_result = self.llm.invoke(messages)

            # Extract the content from the response
            if hasattr(raw_result, 'content'):
                raw_result = raw_result.content
            elif isinstance(raw_result, str):
                pass  # already a string
            else:
                raw_result = str(raw_result)

            # Parse JSON from response
            try:
                import json
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', raw_result, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    result = self._parse_text_to_issues(raw_result, code)
            
            issues = []
            if isinstance(result, list):
                for issue_data in result:
                    try:
                        if isinstance(issue_data, dict):
                            line_num = issue_data.get('line_number', 0)
                            if line_num > 0:
                                code_lines = code.split('\n')
                                if line_num <= len(code_lines):
                                    issue_data['code_snippet'] = code_lines[line_num - 1].strip()
                        
                        issue = CodeIssue(**issue_data)
                        if issue.category == IssueCategory.SECURITY:
                            issues.append(issue)
                    except Exception as e:
                        logger.warning(f"Failed to parse security issue: {e}")
            
            # If no security issues found by AI, check for obvious security issues
            if not issues and code.strip():
                logger.warning("No security issues found by AI, checking for obvious security issues")
                all_obvious = self._detect_obvious_issues(code)
                # Filter only security issues
                security_obvious = [i for i in all_obvious if i.category == IssueCategory.SECURITY]
                issues.extend(security_obvious)
                logger.info(f"Added {len(security_obvious)} obvious security issues")
            
            return issues
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return []
    
    def analyze_performance_focused(self, code: str, language: str) -> List[CodeIssue]:
        """Performance-focused analysis using specialized chain"""
        if not self.llm:
            raise ValueError("No model configured")
        
        # Ensure Langsmith is initialized
        langsmith_client = get_langsmith_client()
        
        try:
            # Create the performance analysis prompt dynamically
            performance_prompt_text = f"""
            You are a performance optimization expert. Analyze performance bottlenecks in this {language} code. Return ONLY valid JSON array.

            CODE TO ANALYZE:
            ```{language}
            {code}
            ```

            You MUST return ONLY a valid JSON array. Find performance issues!

            PERFORMANCE ANALYSIS FOCUS:
            - **Algorithm Complexity**: O(n²) vs O(n log n), inefficient loops, recursive issues
            - **Memory Management**: Memory leaks, excessive allocations, large object retention
            - **I/O Operations**: File I/O bottlenecks, network calls, database queries
            - **CPU Usage**: Heavy computations, blocking operations, thread contention
            - **Resource Management**: Connection pooling, caching opportunities, lazy loading
            - **Scalability Issues**: Hard-coded limits, synchronous operations, single-threaded bottlenecks
            - **Data Structures**: Inefficient data structures, unnecessary copying, redundant operations

            REQUIRED JSON FORMAT:
            [
                {{
                    "severity": "High",
                    "category": "Performance",
                    "line_number": 1,
                    "title": "Specific performance bottleneck",
                    "description": "Detailed explanation of performance impact and root cause",
                    "suggestion": "Concrete optimization steps with expected performance improvement",
                    "confidence": 0.85
                }}
            ]

            QUANTIFY IMPACT: Estimate performance improvement (e.g., "50% faster", "reduces memory by 60%").
            BE SPECIFIC: Include algorithmic complexity analysis and measurable improvements.
            PRIORITIZE: Focus on high-impact performance optimizations.
            NEVER return empty array unless code has NO performance issues at all.
            Your response must be ONLY the JSON array, nothing else.
            """

            # Use the LLM directly with the constructed prompt
            messages = [
                self.system_message,
                HumanMessage(content=performance_prompt_text)
            ]

            raw_result = self.llm.invoke(messages)

            # Extract the content from the response
            if hasattr(raw_result, 'content'):
                raw_result = raw_result.content
            elif isinstance(raw_result, str):
                pass  # already a string
            else:
                raw_result = str(raw_result)

            # Parse JSON from response
            try:
                import json
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', raw_result, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    result = self._parse_text_to_issues(raw_result, code)
            
            issues = []
            if isinstance(result, list):
                for issue_data in result:
                    try:
                        if isinstance(issue_data, dict):
                            line_num = issue_data.get('line_number', 0)
                            if line_num > 0:
                                code_lines = code.split('\n')
                                if line_num <= len(code_lines):
                                    issue_data['code_snippet'] = code_lines[line_num - 1].strip()
                        
                        issue = CodeIssue(**issue_data)
                        if issue.category == IssueCategory.PERFORMANCE:
                            issues.append(issue)
                    except Exception as e:
                        logger.warning(f"Failed to parse performance issue: {e}")
            
            # If no performance issues found by AI, check for obvious performance issues
            if not issues and code.strip():
                logger.warning("No performance issues found by AI, checking for obvious performance issues")
                all_obvious = self._detect_obvious_issues(code)
                # Filter only performance issues
                perf_obvious = [i for i in all_obvious if i.category == IssueCategory.PERFORMANCE]
                issues.extend(perf_obvious)
                logger.info(f"Added {len(perf_obvious)} obvious performance issues")
            
            return issues
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return []
    
    def _analyze_large_code(self, code: str, language: str) -> AnalysisResult:
        """Handle large code files using LangChain text splitter"""
        # Ensure Langsmith is initialized
        langsmith_client = get_langsmith_client()

        start_time = datetime.now()
        
        # Split code into manageable chunks
        documents = [Document(page_content=code)]
        chunks = self.text_splitter.split_documents(documents)
        
        st.info(f"📄 Large codebase detected! Processing {len(chunks)} chunks with {self.current_model}")
        
        all_issues = []
        progress_bar = st.progress(0)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Create the analysis prompt dynamically for this chunk
                chunk_prompt_text = f"""
                You are a code security auditor. Analyze this {language} code chunk VERY CAREFULLY for ANY issues.

                CODE CHUNK TO ANALYZE:
                ```{language}
                {chunk.page_content}
                ```

                You MUST return ONLY a valid JSON array. Find issues in this code chunk!

                REQUIRED JSON FORMAT:
                [
                    {{
                        "severity": "Critical",
                        "category": "Security",
                        "line_number": 1,
                        "title": "Specific security vulnerability",
                        "description": "Detailed explanation of the security risk and impact",
                        "suggestion": "Concrete steps to fix the vulnerability",
                        "confidence": 0.95
                    }}
                ]

                If no issues found in this chunk, return: []

                BE SPECIFIC: Reference actual line numbers and quote problematic code.
                NEVER return empty array unless chunk has NO issues at all.
                Your response must be ONLY the JSON array, nothing else.
                """

                # Use the LLM directly with the constructed prompt
                messages = [
                    self.system_message,
                    HumanMessage(content=chunk_prompt_text)
                ]

                raw_result = self.llm.invoke(messages)

                # Extract the content from the response
                if hasattr(raw_result, 'content'):
                    raw_result = raw_result.content
                elif isinstance(raw_result, str):
                    pass  # already a string
                else:
                    raw_result = str(raw_result)

                # Parse JSON from response
                try:
                    import json
                    result = json.loads(raw_result)
                except json.JSONDecodeError:
                    json_match = re.search(r'\[.*\]', raw_result, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                    else:
                        result = self._parse_text_to_issues(raw_result, chunk.page_content)

                if isinstance(result, list):
                    for issue_data in result:
                        try:
                            if isinstance(issue_data, dict):
                                line_num = issue_data.get('line_number', 0)
                                if line_num > 0:
                                    chunk_lines = chunk.page_content.split('\n')
                                    if line_num <= len(chunk_lines):
                                        issue_data['code_snippet'] = chunk_lines[line_num - 1].strip()
                            
                            issue = CodeIssue(**issue_data)
                            all_issues.append(issue)
                        except Exception as e:
                            logger.warning(f"Failed to parse issue in chunk {i+1}: {e}")
                
                progress_bar.progress((i + 1) / len(chunks))
                
            except Exception as e:
                logger.error(f"Chunk {i+1} analysis failed: {e}")
                continue
        
        progress_bar.empty()
        
        # Generate comprehensive summary
        summary = self._generate_summary(all_issues, code, language)
        quality_score = self._calculate_quality_score(all_issues, len(code.split('\n')))
        recommendations = self._generate_recommendations(all_issues)
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResult(
            issues=all_issues,
            summary=summary,
            quality_score=quality_score,
            model_used=f"{self.current_model} (via OpenRouter)",
            analysis_time=analysis_time,
            recommendations=recommendations,
            token_usage={
                "total_tokens": 0,  # Langsmith tracks this automatically
                "estimated_cost": 0.0  # Langsmith tracks this automatically
            }
        )
    
    def _generate_summary(self, issues: List[CodeIssue], code: str, language: str) -> str:
        """Generate analysis summary using LangChain chain"""
        try:
            # Generate dynamic summary based on actual issues
            if not issues:
                return self._generate_no_issues_summary(language, len(code.split('\n')))

            # Create detailed issues summary
            issues_summary = self._create_issues_summary(issues)

            # Generate dynamic summary using LLM if available
            if self.llm:
                try:
                    # Create the summary prompt dynamically
                    quality_score = self._calculate_quality_score(issues, len(code.split('\n')))
                    summary_prompt_text = f"""
                    Generate a comprehensive executive summary based on this code analysis.

                    CODE ANALYSIS DATA:
                    - Programming Language: {language}
                    - Lines of Code: {len(code.split('\n'))}
                    - Total Issues Found: {len(issues)}
                    - AI Model Used: {self.current_model}
                    - Quality Score: {quality_score}/100

                    ISSUES SUMMARY:
                    {issues_summary}

                    Write a professional executive summary that includes:

                    1. **Overall Assessment**: Evaluate the code quality based on the issues found
                    2. **Critical Issues**: Highlight any immediate security or stability concerns
                    3. **Priority Actions**: Recommend the top 3 most important fixes needed
                    4. **Quality Score Explanation**: Explain why the code received this quality score
                    5. **Improvement Strategy**: Suggest a roadmap for enhancing code quality

                    IMPORTANT: Use the actual values provided above. Do NOT use placeholder text like {{language}} or {{total_issues}}. Replace them with the real values.

                    Format your response as a clean, professional summary without any code blocks or markdown artifacts.
                    """

                    # Use the LLM directly with the constructed prompt
                    messages = [
                        self.system_message,
                        HumanMessage(content=summary_prompt_text)
                    ]

                    summary_result = self.llm.invoke(messages)

                    # Extract the content from the response
                    if hasattr(summary_result, 'content'):
                        summary = summary_result.content
                    elif isinstance(summary_result, str):
                        summary = summary_result
                    else:
                        summary = str(summary_result)

                    # Clean any remaining placeholders from LLM response
                    cleaned_summary = self._clean_summary_placeholders(
                        summary, language, len(code.split('\n')), len(issues), self.current_model
                    )
                    return cleaned_summary
                except Exception as e:
                    logger.warning(f"LLM summary generation failed: {e}")
                    return self._generate_fallback_summary(issues, language, len(code.split('\n')))

            # Fallback summary generation
            return self._generate_fallback_summary(issues, language, len(code.split('\n')))
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return self._generate_error_summary(issues, language, len(code.split('\n')))

    def _generate_no_issues_summary(self, language: str, lines_of_code: int) -> str:
        """Generate summary when no issues are found"""
        return f"""## Executive Summary

**Analysis Results:** Comprehensive code review completed successfully. No significant issues were identified in the {language} codebase.

**Code Statistics:**
- Language: {language}
- Lines of code: {lines_of_code:,}
- Total issues found: 0
- Model used: {self.current_model}

**Overall Assessment:** The codebase demonstrates excellent coding practices with strong attention to security, performance, and maintainability. The code structure is clean and well-organized.

**Critical Issues:** None identified - no immediate security or stability concerns.

**Priority Actions:**
1. Continue maintaining current coding standards
2. Regular code reviews to maintain quality
3. Keep dependencies updated

**Quality Score Justification:** Score of 95/100 reflects the high quality of the codebase with no significant issues found and adherence to best practices.

**Improvement Roadmap:** Continue with current development practices and consider implementing automated testing if not already in place."""

    def _generate_fallback_summary(self, issues: List[CodeIssue], language: str, lines_of_code: int) -> str:
        """Generate fallback summary without LLM"""
        severity_counts = {}
        category_counts = {}

        for issue in issues:
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
            category_counts[issue.category.value] = category_counts.get(issue.category.value, 0) + 1

        critical_issues = severity_counts.get('Critical', 0)
        high_issues = severity_counts.get('High', 0)
        quality_score = self._calculate_quality_score(issues, lines_of_code)

        if critical_issues > 0:
            assessment = f"The {language} codebase requires immediate attention with {critical_issues} critical issues that may impact security and stability."
        elif high_issues > 0:
            assessment = f"The {language} codebase has {high_issues} high-priority issues that should be addressed in the next development cycle."
        else:
            assessment = f"The {language} codebase shows good quality with only minor issues that don't require immediate attention."

        # Generate priority actions based on issues
        priority_actions = []
        if critical_issues > 0:
            priority_actions.append("Address all critical security and stability issues immediately")
        if high_issues > 0:
            priority_actions.append("Fix high-priority issues in the next development sprint")
        priority_actions.append("Plan maintenance for medium and low-priority improvements")

        return f"""**Analysis Results:** Code review completed for {language} codebase ({lines_of_code:,} lines) identified {len(issues)} issues requiring attention across security, performance, and maintainability aspects.

**Overall Assessment:** {assessment}

**Critical Issues:** {critical_issues} critical issues found that require immediate attention.
**High Priority:** {high_issues} high-priority issues identified.

**Priority Actions:**
{"\n".join(f"{i+1}. {action}" for i, action in enumerate(priority_actions))}

**Quality Score Justification:** The codebase received a quality score of {quality_score}/100 based on {len(issues)} issues found across {lines_of_code:,} lines of code, with {critical_issues} critical and {high_issues} high-priority issues affecting the overall assessment.

**Improvement Roadmap:** Focus on resolving critical and high-priority issues first, followed by systematic improvement of code quality and implementation of best practices."""

    def _generate_error_summary(self, issues: List[CodeIssue], language: str, lines_of_code: int) -> str:
        """Generate summary when there's an error"""
        quality_score = self._calculate_quality_score(issues, lines_of_code)
        return f"""**Analysis Results:** Code review completed for {language} codebase ({lines_of_code:,} lines) with some technical difficulties. Found {len(issues)} issues in the analysis.

**Overall Assessment:** Analysis completed successfully despite technical challenges. Please review the detailed findings below.

**Critical Issues:** {len([i for i in issues if i.severity == Severity.CRITICAL])} critical issues identified.
**High Priority:** {len([i for i in issues if i.severity == Severity.HIGH])} high-priority issues found.

**Quality Score:** The codebase received a quality score of {quality_score}/100 based on available analysis results.

**Recommendations:** Review all identified issues and implement fixes as needed.

**Note:** Some advanced analysis features may have been limited due to technical constraints."""

    def _create_issues_summary(self, issues: List[CodeIssue]) -> str:
        """Create a formatted summary of issues"""
        if not issues:
            return "No issues found"

        # Group issues by severity and category
        severity_groups = {}
        category_groups = {}

        for issue in issues[:15]:  # Limit for summary
            severity = issue.severity.value
            category = issue.category.value

            if severity not in severity_groups:
                severity_groups[severity] = []
            if category not in category_groups:
                category_groups[category] = []

            severity_groups[severity].append(issue)
            category_groups[category].append(issue)

        summary_lines = []

        # Add critical issues first
        if 'Critical' in severity_groups:
            summary_lines.append("**Critical Issues:**")
            for issue in severity_groups['Critical']:
                summary_lines.append(f"- Line {issue.line_number}: {issue.title}")

        # Add high priority issues
        if 'High' in severity_groups:
            summary_lines.append("\n**High Priority Issues:**")
            for issue in severity_groups['High']:
                summary_lines.append(f"- Line {issue.line_number}: {issue.title}")

        # Add category breakdown
        summary_lines.append(f"\n**Issues by Category:** {', '.join(f'{cat}: {len(category_groups[cat])}' for cat in category_groups)}")

        return "\n".join(summary_lines)
    
    def _calculate_quality_score(self, issues: List[CodeIssue], lines_of_code: int) -> float:
        """Calculate code quality score based on issues found"""
        if not issues:
            return 95.0
        
        # Enhanced severity-based penalties with more realistic scoring
        severity_weights = {
            Severity.CRITICAL: 30,  # Higher penalty for critical issues
            Severity.HIGH: 20,      # Higher penalty for high issues
            Severity.MEDIUM: 10,    # Moderate penalty for medium issues
            Severity.LOW: 5         # Lower penalty for low issues
        }

        # Category-based additional penalties (security issues are more severe)
        category_multipliers = {
            IssueCategory.SECURITY: 1.5,      # Security issues are 50% more severe
            IssueCategory.PERFORMANCE: 1.2,   # Performance issues are 20% more severe
            IssueCategory.BUG: 1.3,           # Bugs are 30% more severe
            IssueCategory.STYLE: 1.0,         # Style issues have normal weight
            IssueCategory.MAINTAINABILITY: 1.1  # Maintainability issues are slightly more severe
        }

        total_penalty = 0
        for issue in issues:
            base_weight = severity_weights.get(issue.severity, 10)
            category_multiplier = category_multipliers.get(issue.category, 1.0)
            total_penalty += base_weight * category_multiplier

        # Adjust for code size - normalize by lines of code
        # This prevents large codebases from getting unfairly low scores
        size_factor = min(1.0, 1000 / max(lines_of_code, 100))  # Cap at 1000 lines for normalization
        normalized_penalty = total_penalty * size_factor

        # Calculate final score with minimum bounds
        base_score = 100
        final_score = max(10, base_score - normalized_penalty)  # Minimum score of 10
        
        return round(final_score, 1)
    
    def _parse_text_to_issues(self, text: str, code: str) -> List[CodeIssue]:
        """Parse text response into CodeIssue objects when JSON parsing fails"""
        issues = []

        # If no response or very short response, return empty list
        if not text or len(text.strip()) < 20:
            return issues

        # Try to extract JSON from the response if it's wrapped in text
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',  # JSON in code blocks
            r'```\s*(\[.*?\])\s*```',      # JSON in code blocks without json tag
            r'(\[.*\])',                    # Just the JSON array
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        result = json.loads(match)
                        if isinstance(result, list):
                            for issue_data in result:
                                if isinstance(issue_data, dict):
                                    try:
                                        # Add code snippet if line number is valid
                                        line_num = issue_data.get('line_number', 0)
                                        if line_num > 0:
                                            code_lines = code.split('\n')
                                            if line_num <= len(code_lines):
                                                issue_data['code_snippet'] = code_lines[line_num - 1].strip()

                                        issue = CodeIssue(**issue_data)
                                        issues.append(issue)
                                    except Exception as e:
                                        logger.warning(f"Failed to parse issue: {e}")
                                        continue
                    except json.JSONDecodeError:
                        continue

        # If no JSON found, try to parse as structured text
        if not issues:
            issues = self._parse_structured_text(text, code)

        return issues

    def _parse_structured_text(self, text: str, code: str) -> List[CodeIssue]:
        """Parse structured text response into CodeIssue objects"""
        issues = []
        code_lines = code.split('\n')

        # Split by common issue separators
        issue_blocks = re.split(r'\n\s*(?=\d+\.|\*\s*|\-\s*[A-Z])', text)

        for block in issue_blocks:
            block = block.strip()
            if not block or len(block) < 10:
                continue
            
            # Try to extract issue information
            issue_info = self._extract_issue_info(block, code_lines)
            if issue_info:
                try:
                    issues.append(CodeIssue(**issue_info))
                except Exception as e:
                    logger.warning(f"Failed to create issue from text: {e}")
                    continue

        return issues

    def _extract_issue_info(self, text: str, code_lines: list) -> Optional[dict]:
        """Extract issue information from text block"""
        # Determine severity
        severity = Severity.MEDIUM
        if any(word in text.upper() for word in ['CRITICAL', 'SEVERE', 'URGENT', 'DANGER']):
            severity = Severity.CRITICAL
        elif any(word in text.upper() for word in ['HIGH', 'IMPORTANT']):
            severity = Severity.HIGH
        elif any(word in text.upper() for word in ['LOW', 'MINOR', 'NIT']):
            severity = Severity.LOW

        # Determine category
        category = IssueCategory.MAINTAINABILITY
        if any(word in text.upper() for word in ['SECURITY', 'VULNERABLE', 'ATTACK', 'INJECTION', 'XSS', 'CSRF']):
            category = IssueCategory.SECURITY
        elif any(word in text.upper() for word in ['PERFORMANCE', 'SLOW', 'OPTIMIZE', 'MEMORY', 'SPEED']):
            category = IssueCategory.PERFORMANCE
        elif any(word in text.upper() for word in ['BUG', 'ERROR', 'CRASH', 'FAIL', 'NULL', 'EXCEPTION']):
            category = IssueCategory.BUG
        elif any(word in text.upper() for word in ['STYLE', 'FORMAT', 'PEP', 'NAMING', 'CONVENTION']):
            category = IssueCategory.STYLE
                
                # Extract line number
        line_num = 0
        line_matches = re.findall(r'line\s*(\d+)', text, re.IGNORECASE)
        if line_matches:
            line_num = int(line_matches[0])

        # Extract title and description
        title = text[:60] + "..." if len(text) > 60 else text
        description = text

        # Extract suggestion if present
        suggestion = "Review and fix this issue"
        if "suggestion:" in text.lower() or "fix:" in text.lower() or "recommend:" in text.lower():
            parts = re.split(r'suggestion:|fix:|recommend:', text, flags=re.IGNORECASE)
            if len(parts) > 1:
                suggestion = parts[1].strip()[:200]

        # Add code snippet if line number is valid
        code_snippet = ""
        if line_num > 0 and line_num <= len(code_lines):
            code_snippet = code_lines[line_num - 1].strip()

        return {
            'severity': severity,
            'category': category,
            'line_number': line_num,
            'title': title,
            'description': description,
            'suggestion': suggestion,
            'code_snippet': code_snippet,
            'confidence': 0.8
        }
    
    def _parse_analysis_response(self, raw_result: str, code: str) -> List[dict]:
        """Enhanced parsing of analysis response with multiple fallback strategies"""
        # Clean the raw result first
        cleaned_result = raw_result.strip()
        logger.info(f"Parsing response: '{cleaned_result[:200]}...'")

        # Strategy 1: Try direct JSON parsing
        try:
            result = json.loads(cleaned_result)
            if isinstance(result, list):
                logger.info(f"Strategy 1 success: Found {len(result)} issues")
                return result
        except json.JSONDecodeError as e:
            logger.info(f"Strategy 1 failed: {e}")

        # Strategy 2: Extract JSON from code blocks
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',  # JSON in code blocks
            r'```\s*(\[.*?\])\s*```',      # JSON in code blocks without json tag
            r'```(\[.*?\])```',            # JSON in code blocks (alternative)
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned_result, re.DOTALL)
            logger.info(f"Strategy 2 pattern '{pattern}' found {len(matches)} matches")
            for match in matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, list):
                        logger.info(f"Strategy 2 success: Found {len(result)} issues")
                        return result
                except json.JSONDecodeError as e:
                    logger.info(f"Strategy 2 JSON decode failed: {e}")
                    continue

        # Strategy 3: Extract any JSON array from the text
        json_array_patterns = [
            r'(\[[\s\S]*?\])',  # Any JSON array (non-greedy)
        ]

        for pattern in json_array_patterns:
            matches = re.findall(pattern, cleaned_result)
            logger.info(f"Strategy 3 pattern found {len(matches)} matches")
            for match in matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, list):
                        logger.info(f"Strategy 3 success: Found {len(result)} issues")
                        return result
                except json.JSONDecodeError as e:
                    logger.info(f"Strategy 3 JSON decode failed: {e}")
                    continue

        # Strategy 4: Try to find structured content and parse as text
        logger.warning("All JSON parsing strategies failed, attempting text-based parsing")
        text_result = self._parse_text_to_issues(cleaned_result, code)
        logger.info(f"Text parsing result: {len(text_result)} issues")
        return text_result
    
    def _create_issue_from_dict(self, issue_dict: dict, code_lines: list) -> Optional[CodeIssue]:
        """Create CodeIssue from dictionary with validation"""
        try:
            # Add code snippet if line number is valid
            line_num = issue_dict.get('line_number', 0)
            if line_num > 0 and line_num <= len(code_lines):
                issue_dict['code_snippet'] = code_lines[line_num - 1].strip()
            else:
                issue_dict['code_snippet'] = ""
            
            return CodeIssue(**issue_dict)
        except Exception as e:
            logger.warning(f"Failed to create issue: {e}")
            return None

    def _generate_recommendations(self, issues: List[CodeIssue]) -> List[str]:
        """Generate prioritized recommendations based on issues"""
        recommendations = []
        
        if not issues:
            return ["✅ **EXCELLENT**: Code quality is high, continue following current practices",
                    "🔍 **MAINTENANCE**: Consider implementing automated testing and code review processes",
                    "📚 **BEST PRACTICES**: Keep dependencies updated and follow security guidelines"]
        
        # Categorize issues
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        medium_issues = [i for i in issues if i.severity == Severity.MEDIUM]
        low_issues = [i for i in issues if i.severity == Severity.LOW]

        security_issues = [i for i in issues if i.category == IssueCategory.SECURITY]
        performance_issues = [i for i in issues if i.category == IssueCategory.PERFORMANCE]
        bug_issues = [i for i in issues if i.category == IssueCategory.BUG]
        style_issues = [i for i in issues if i.category == IssueCategory.STYLE]
        maintainability_issues = [i for i in issues if i.category == IssueCategory.MAINTAINABILITY]
        
        # Generate contextual recommendations with specific priorities
        if critical_issues:
            recommendations.append(f"🚨 **CRITICAL URGENCY**: Address {len(critical_issues)} critical issues immediately - these pose significant security/stability risks")
            if security_issues:
                recommendations.append(f"🔒 **SECURITY FIRST**: Prioritize {len(security_issues)} security vulnerabilities - deploy security patches ASAP")
            if bug_issues:
                recommendations.append(f"🐛 **BUG FIXES**: Resolve {len(bug_issues)} bugs that may cause system crashes or incorrect behavior")
        
        if high_issues:
            recommendations.append(f"⚠️ **HIGH PRIORITY**: Fix {len(high_issues)} high-severity issues in the next development sprint")
            if performance_issues:
                recommendations.append(f"⚡ **PERFORMANCE**: Optimize {len(performance_issues)} performance bottlenecks affecting user experience")

        if medium_issues:
            recommendations.append(f"📋 **MEDIUM TERM**: Plan to address {len(medium_issues)} medium-priority issues within 2-3 sprints")

        if low_issues:
            recommendations.append(f"🔧 **MAINTENANCE**: Schedule {len(low_issues)} low-priority improvements for future releases")

        # Add category-specific recommendations
        if maintainability_issues:
            recommendations.append(f"🏗️ **CODE ARCHITECTURE**: Refactor {len(maintainability_issues)} maintainability issues to improve long-term code health")

        if style_issues:
            recommendations.append(f"💅 **CODE STYLE**: Apply {len(style_issues)} style improvements to ensure consistent coding standards")

        # Add general recommendations based on issue density
        if len(issues) > 50:
            recommendations.append("🏛️ **ARCHITECTURAL REVIEW**: High issue count suggests considering code restructuring or architectural improvements")
        elif len(issues) > 20:
            recommendations.append("🔍 **CODE REVIEW PROCESS**: Consider implementing peer code reviews to maintain quality standards")
        else:
            recommendations.append("✅ **QUALITY MAINTENANCE**: Continue current development practices while addressing identified issues")

        # Add specific actionable advice
        if security_issues:
            recommendations.append("🛡️ **SECURITY TESTING**: Implement automated security scanning in CI/CD pipeline")
        
        if performance_issues:
            recommendations.append("📊 **PERFORMANCE MONITORING**: Set up performance monitoring and establish performance benchmarks")

        if bug_issues:
            recommendations.append("🧪 **TESTING STRATEGY**: Enhance unit and integration test coverage to catch bugs early")

        return recommendations[:8]  # Limit to top 8 recommendations

    def _detect_obvious_issues(self, code: str) -> List[CodeIssue]:
        """Detect obvious security and code quality issues that should always be found"""
        issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            line_clean = line.strip()

            # Check for hardcoded API keys/secrets
            if any(pattern in line_clean.upper() for pattern in ['API_KEY =', 'SECRET =', 'PASSWORD =', 'TOKEN =']):
                # Look for quotes indicating hardcoded values
                if ('"' in line_clean or "'" in line_clean) and not line_clean.upper().endswith('= NONE'):
                    issues.append(CodeIssue(
                        severity=Severity.CRITICAL,
                        category=IssueCategory.SECURITY,
                        line_number=i,
                        title="Hardcoded Secret Detected",
                        description="API key, password, or secret appears to be hardcoded in the source code. This is a major security vulnerability.",
                        suggestion="Move secrets to environment variables or a secure configuration management system.",
                        code_snippet=line_clean,
                        confidence=0.95
                    ))

            # Check for debug mode in production
            if 'DEBUG=TRUE' in line_clean.upper() or 'DEBUG = TRUE' in line_clean.upper():
                issues.append(CodeIssue(
                    severity=Severity.HIGH,
                    category=IssueCategory.SECURITY,
                    line_number=i,
                    title="Debug Mode Enabled in Production",
                    description="Debug mode is enabled, which can expose sensitive information and reduce performance in production.",
                    suggestion="Set debug=False for production deployments and use proper logging instead.",
                    code_snippet=line_clean,
                    confidence=0.9
                ))

            # Check for CORS allowing all origins
            if 'ALLOW_ORIGINS' in line_clean.upper() and ('["*"]' in line_clean or '["*"]' in line_clean):
                issues.append(CodeIssue(
                    severity=Severity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    line_number=i,
                    title="CORS Allows All Origins",
                    description="CORS is configured to allow all origins (*), which is a security vulnerability.",
                    suggestion="Specify allowed origins explicitly instead of using wildcard (*).",
                    code_snippet=line_clean,
                    confidence=0.95
                ))

            # Check for no authentication on sensitive endpoints
            if ('@app.' in line_clean or '@router.' in line_clean) and 'admin' in line_clean.lower():
                # Check next few lines for auth
                auth_found = False
                for j in range(min(5, len(lines) - i)):
                    next_line = lines[i + j].strip()
                    if any(auth in next_line.upper() for auth in ['AUTH', 'JWT', 'TOKEN', 'LOGIN', 'PASSWORD']):
                        auth_found = True
                        break

                if not auth_found:
                    issues.append(CodeIssue(
                        severity=Severity.CRITICAL,
                        category=IssueCategory.SECURITY,
                        line_number=i,
                        title="Admin Endpoint Without Authentication",
                        description="Admin endpoint detected without visible authentication checks.",
                        suggestion="Add proper authentication and authorization to admin endpoints.",
                        code_snippet=line_clean,
                        confidence=0.9
                    ))

            # Check for logging sensitive data
            if 'print(' in line_clean and any(sensitive in line_clean.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                issues.append(CodeIssue(
                    severity=Severity.HIGH,
                    category=IssueCategory.SECURITY,
                    line_number=i,
                    title="Logging Sensitive Data",
                    description="Code appears to be logging or printing sensitive information like passwords or tokens.",
                    suggestion="Remove logging of sensitive data. Use proper logging with data masking if needed.",
                    code_snippet=line_clean,
                    confidence=0.85
                ))

            # Check for host 0.0.0.0 (security risk)
            if 'host="0.0.0.0"' in line_clean or "host='0.0.0.0'" in line_clean:
                issues.append(CodeIssue(
                    severity=Severity.MEDIUM,
                    category=IssueCategory.SECURITY,
                    line_number=i,
                    title="Server Bound to All Interfaces",
                    description="Server is bound to 0.0.0.0, exposing it to all network interfaces.",
                    suggestion="Bind to specific interfaces (127.0.0.1 for local development, specific IP for production).",
                    code_snippet=line_clean,
                    confidence=0.8
                ))

        logger.info(f"Detected {len(issues)} obvious issues")
        return issues

    def _clean_summary_placeholders(self, summary: str, language: str, loc: int, total_issues: int, model_name: str) -> str:
        """Clean any remaining placeholders in the summary text"""
        # Replace any remaining placeholders that the LLM might have left
        replacements = {
            '{language}': language,
            '{loc}': str(loc),
            '{total_issues}': str(total_issues),
            '{model_name}': model_name,
            '{{language}}': language,
            '{{loc}}': str(loc),
            '{{total_issues}}': str(total_issues),
            '{{model_name}}': model_name,
            '{score}': str(self._calculate_quality_score([], loc))  # Default score if not calculated
        }

        cleaned_summary = summary
        for placeholder, value in replacements.items():
            cleaned_summary = cleaned_summary.replace(placeholder, value)

        return cleaned_summary

def main():
    st.set_page_config(
        page_title="LangChain + OpenRouter Code Review",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 LangChain Multi-Model Code Review")
    st.caption("**Powered by OpenRouter API** - Access 20+ AI models through LangChain")
    
    # Model showcase info
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.info("🤖 **12+ Models** - Claude, GPT-4, Gemini, Llama")
    with col_info2:
        st.info("🔗 **LangChain Powered** - Structured chains & parsing")
    with col_info3:
        st.info("💰 **Cost Tracking** - Monitor usage across models")
    
    # Initialize session state
    if 'reviewer' not in st.session_state:
        st.session_state.reviewer = OpenRouterLangChainReviewer()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    reviewer = st.session_state.reviewer
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔑 OpenRouter Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your API key from https://openrouter.ai/keys"
        )
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Enter OpenRouter API key to continue")
        
        st.divider()
        
        # Model selection with detailed info
        st.header("🤖 Model Selection")
        
        # Group models by provider
        providers = {}
        for model_name, info in OPENROUTER_MODELS.items():
            provider = info["provider"]
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model_name)
        
        # Provider selection
        selected_provider = st.selectbox(
            "AI Provider",
            list(providers.keys()),
            help="Choose your preferred AI provider"
        )
        
        # Model selection within provider
        available_models = providers[selected_provider]
        selected_model = st.selectbox(
            "Model",
            available_models,
            help="Select specific model variant"
        )
        
        # Show model details
        if selected_model:
            model_info = reviewer.get_model_info(selected_model)
            st.info(f"**Context:** {model_info.get('context', 'Unknown')}")
            st.info(f"**Notes:** {model_info.get('cost', 'Standard pricing')}")
        
        # Model setup
        if api_key and selected_model:
            if st.button("🚀 Initialize Model"):
                with st.spinner(f"Setting up {selected_model}..."):
                    success = reviewer.setup_model(
                        model_name=selected_model,
                        api_key=api_key,
                        temperature=0.1
                    )
                    
                    if success:
                        st.success(f"✅ {selected_model} ready!")
                        st.session_state.model_ready = True
                    else:
                        st.error("❌ Model setup failed")
                        st.session_state.model_ready = False
        
        # Show active model
        if reviewer.current_model:
            st.success(f"🤖 **Active:** {reviewer.current_model}")
            
            # Usage statistics
            if reviewer.callback.model_calls:
                st.metric("API Calls", len(reviewer.callback.model_calls))
                st.metric("Total Tokens", f"{reviewer.callback.total_tokens:,}")
                if reviewer.callback.total_cost > 0:
                    st.metric("Est. Cost", f"${reviewer.callback.total_cost:.4f}")
        
        st.divider()
        
        # Analysis configuration
        st.header("📊 Analysis Settings")
        
        language = st.selectbox(
            "Programming Language",
            ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", 
             "PHP", "C#", "Swift", "Kotlin", "Ruby", "Scala", "SQL"],
            help="Select the primary language of your code"
        )
        
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["🔍 Comprehensive", "🔒 Security Focused", "⚡ Performance Focused"],
            help="Choose analysis depth and focus area"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            temperature = st.slider(
                "Model Temperature", 
                0.0, 1.0, 0.1, 0.1,
                help="Lower = more focused, Higher = more creative"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.6, 0.1,
                help="Filter out low-confidence issues"
            )
            
            max_issues_display = st.selectbox(
                "Max Issues to Display",
                [10, 25, 50, 100, "All"],
                index=2
            )
            
            chunk_size = st.slider(
                "Chunk Size (for large files)",
                2000, 8000, 4000, 500,
                help="Smaller chunks = more detailed analysis"
            )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Code Input")
        
        # Input method tabs
        tab1, tab2, tab3 = st.tabs(["📝 Text Editor", "📁 File Upload", "📂 Multiple Files"])
        
        code_content = ""
        file_info = {}
        
        with tab1:
            code_content = st.text_area(
                "Enter your code:",
                height=400,
                placeholder=f"""// Paste your {language} code here
// LangChain + OpenRouter will analyze it with advanced AI models

def example_function():
    # Your code here
    pass""",
                help="Paste code directly for immediate analysis"
            )
        
        with tab2:
            uploaded_file = st.file_uploader(
                "Upload single code file",
                type=['py', 'js', 'ts', 'java', 'cpp', 'go', 'rs', 'php', 'cs', 'rb', 'scala', 'kt', 'swift', 'txt'],
                help="Upload a code file for analysis"
            )
            
            if uploaded_file:
                code_content = str(uploaded_file.read(), "utf-8")
                lines = len(code_content.split('\n'))
                chars = len(code_content)
                
                file_info = {
                    'name': uploaded_file.name,
                    'lines': lines,
                    'size': chars,
                    'estimated_tokens': chars // 4
                }
                
                st.success(f"📁 **{uploaded_file.name}** loaded successfully!")
                
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    st.metric("Lines", f"{lines:,}")
                with col_f2:
                    st.metric("Characters", f"{chars:,}")
                with col_f3:
                    st.metric("Est. Tokens", f"{chars//4:,}")
                
                with st.expander("📄 File Preview"):
                    preview_length = min(2000, len(code_content))
                    st.code(
                        code_content[:preview_length] + ("..." if len(code_content) > preview_length else ""),
                        language=language.lower()
                    )
        
        with tab3:
            uploaded_files = st.file_uploader(
                "Upload multiple code files",
                type=['py', 'js', 'ts', 'java', 'cpp', 'go', 'rs', 'php', 'cs', 'rb', 'scala', 'kt', 'swift', 'txt'],
                accept_multiple_files=True,
                help="Upload multiple files to analyze as a combined codebase"
            )
            
            if uploaded_files:
                combined_code = []
                total_lines = 0
                total_chars = 0
                
                for file in uploaded_files:
                    file_content = str(file.read(), "utf-8")
                    file_lines = len(file_content.split('\n'))
                    
                    combined_code.append(f"""
# ===== FILE: {file.name} =====
# Lines: {file_lines}
{file_content}
""")
                    total_lines += file_lines
                    total_chars += len(file_content)
                
                code_content = "\n".join(combined_code)
                
                file_info = {
                    'files': len(uploaded_files),
                    'total_lines': total_lines,
                    'total_chars': total_chars,
                    'estimated_tokens': total_chars // 4
                }
                
                st.success(f"📂 **{len(uploaded_files)} files** combined successfully!")
                
                col_mf1, col_mf2, col_mf3 = st.columns(3)
                with col_mf1:
                    st.metric("Files", len(uploaded_files))
                with col_mf2:
                    st.metric("Total Lines", f"{total_lines:,}")
                with col_mf3:
                    st.metric("Est. Tokens", f"{total_chars//4:,}")
                
                with st.expander("📋 Files Included"):
                    for file in uploaded_files:
                        st.write(f"• {file.name}")
        
        # Analysis execution
        can_analyze = (code_content and 
                      reviewer.current_model and 
                      api_key)
        
        # Dynamic button text
        if not api_key:
            button_text = "🔑 Enter OpenRouter API Key First"
        elif not reviewer.current_model:
            button_text = "🤖 Initialize Model First"
        elif not code_content:
            button_text = "📝 Add Code to Analyze"
        else:
            estimated_tokens = len(code_content) // 4
            if estimated_tokens > 50000:
                button_text = f"🚀 Analyze Large Codebase ({estimated_tokens:,} tokens)"
            else:
                button_text = f"🚀 Start {analysis_mode.split()[1]} Analysis"
        
        if st.button(button_text, type="primary", disabled=not can_analyze):
            with st.spinner(f"🔍 Running {analysis_mode} with {reviewer.current_model}..."):
                try:
                    # Update model temperature if changed
                    if hasattr(reviewer.llm, 'temperature'):
                        reviewer.llm.temperature = temperature
                    
                    # Run appropriate analysis
                    if analysis_mode == "🔍 Comprehensive":
                        result = reviewer.analyze_code_comprehensive(code_content, language)
                        
                    elif analysis_mode == "🔒 Security Focused":
                        issues = reviewer.analyze_security_focused(code_content, language)
                        
                        # Create result object for security analysis
                        result = AnalysisResult(
                            issues=issues,
                            summary=f"Security analysis completed with {reviewer.current_model}. Found {len(issues)} potential security vulnerabilities requiring attention.",
                            quality_score=max(50, 100 - len([i for i in issues if i.severity == Severity.CRITICAL]) * 15 - len([i for i in issues if i.severity == Severity.HIGH]) * 8),
                            model_used=f"{reviewer.current_model} (Security Focus)",
                            analysis_time=1.0,
                            recommendations=[f"🔒 Review and fix {len(issues)} security issues"],
                            token_usage={
                                "total_tokens": reviewer.callback.total_tokens,
                                "estimated_cost": round(reviewer.callback.total_cost, 4)
                            }
                        )
                        
                    else:  # Performance Focused
                        issues = reviewer.analyze_performance_focused(code_content, language)
                        
                        result = AnalysisResult(
                            issues=issues,
                            summary=f"Performance analysis completed with {reviewer.current_model}. Identified {len(issues)} optimization opportunities to improve application performance.",
                            quality_score=max(60, 100 - len([i for i in issues if i.severity == Severity.HIGH]) * 10 - len([i for i in issues if i.severity == Severity.MEDIUM]) * 5),
                            model_used=f"{reviewer.current_model} (Performance Focus)",
                            analysis_time=1.0,
                            recommendations=[f"⚡ Optimize {len(issues)} performance bottlenecks"],
                            token_usage={
                                "total_tokens": reviewer.callback.total_tokens,
                                "estimated_cost": round(reviewer.callback.total_cost, 4)
                            }
                        )
                    
                    # Apply confidence filtering
                    if confidence_threshold > 0:
                        original_count = len(result.issues)
                        result.issues = [i for i in result.issues if i.confidence >= confidence_threshold]
                        filtered_count = len(result.issues)
                        
                        if filtered_count < original_count:
                            st.info(f"🎯 Filtered {original_count - filtered_count} low-confidence issues (< {confidence_threshold})")
                    
                    # Limit display results
                    if max_issues_display != "All" and len(result.issues) > int(max_issues_display):
                        result.issues = result.issues[:int(max_issues_display)]
                        st.info(f"📊 Showing top {max_issues_display} issues (use sidebar to show more)")
                    
                    st.session_state.analysis_results = result
                    st.session_state.original_code = code_content
                    st.session_state.file_info = file_info
                    
                    # Success message with metrics
                    success_msg = f"✅ **Analysis Complete!** Found {len(result.issues)} issues"
                    if result.token_usage.get('total_tokens'):
                        success_msg += f" • Used {result.token_usage['total_tokens']:,} tokens"
                    if result.token_usage.get('estimated_cost'):
                        success_msg += f" • Est. cost: ${result.token_usage['estimated_cost']:.4f}"
                    
                    st.success(success_msg)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Try a different model or check your API key")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if st.session_state.analysis_results:
            result = st.session_state.analysis_results
            
            # Quality score with color coding
            quality_score = result.quality_score
            if quality_score >= 85:
                score_color = "🟢"
            elif quality_score >= 70:
                score_color = "🟡"
            elif quality_score >= 50:
                score_color = "🟠"
            else:
                score_color = "🔴"

            # Display metrics in large, vertical format
            col_metric1, col_metric2 = st.columns([1, 1])

            with col_metric1:
                # Quality Score - BIG
                st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>Quality Score</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; font-size: 48px;'>{score_color} {quality_score:.1f}/100</h1>", unsafe_allow_html=True)

                # Critical Issues - BIG
                critical_count = len([i for i in result.issues if i.severity == Severity.CRITICAL])
                st.markdown("<h2 style='text-align: center; color: #d62728;'>Critical Issues</h2>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; font-size: 42px; color: #d62728;'>{critical_count}</h1>", unsafe_allow_html=True)

            with col_metric2:
                # High Priority - BIG
                high_count = len([i for i in result.issues if i.severity == Severity.HIGH])
                st.markdown("<h2 style='text-align: center; color: #ff7f0e;'>High Priority</h2>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; font-size: 42px; color: #ff7f0e;'>{high_count}</h1>", unsafe_allow_html=True)

                # Total Issues - BIG
                st.markdown("<h2 style='text-align: center; color: #2ca02c;'>Total Issues</h2>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; font-size: 42px; color: #2ca02c;'>{len(result.issues)}</h1>", unsafe_allow_html=True)

            st.markdown("---")

            # Model and time info
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"🤖 **Model:** {result.model_used}")
            with col_info2:
                st.markdown(f"⏱️ **Time:** {result.analysis_time:.1f}s")
            
            # Token usage info
            if result.token_usage:
                col_usage1, col_usage2 = st.columns(2)
                with col_usage1:
                    if result.token_usage.get('total_tokens'):
                        st.info(f"🔢 **Tokens:** {result.token_usage['total_tokens']:,}")
                with col_usage2:
                    if result.token_usage.get('estimated_cost'):
                        st.info(f"💰 **Est. Cost:** ${result.token_usage['estimated_cost']:.4f}")
            
            # Executive summary
            st.markdown("### 📋 Executive Summary")
            st.markdown(result.summary)
            
            # Key recommendations
            if result.recommendations:
                st.markdown("### 💡 Priority Recommendations")
                for i, rec in enumerate(result.recommendations):
                    if i < 5:  # Limit to top 5 recommendations
                        st.markdown(rec)
            
            # Issues visualization
            if result.issues:
                st.subheader("📊 Issues Breakdown")
                
                # Create severity and category charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    severity_counts = {}
                    for issue in result.issues:
                        severity = issue.severity.value
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    if severity_counts:
                        st.bar_chart(
                            severity_counts,
                            use_container_width=True,
                            height=200
                        )
                        st.caption("Issues by Severity")
                
                with col_chart2:
                    category_counts = {}
                    for issue in result.issues:
                        category = issue.category.value
                        category_counts[category] = category_counts.get(category, 0) + 1
                    
                    if category_counts:
                        st.bar_chart(
                            category_counts,
                            use_container_width=True,
                            height=200
                        )
                        st.caption("Issues by Category")
        
        else:
            st.info("👆 Configure OpenRouter, select a model, and analyze code to see results")
            
            # Feature showcase
            with st.expander("🌟 LangChain + OpenRouter Features"):
                st.markdown("""
                **🔗 LangChain Integration:**
                - Structured prompt templates with ChatPromptTemplate
                - Type-safe parsing with Pydantic models
                - Chain composition using LCEL (LangChain Expression Language)
                - Automatic text splitting for large codebases
                - Memory and conversation tracking
                - Custom callbacks for usage monitoring
                
                **🤖 Multi-Model Support via OpenRouter:**
                - **Anthropic:** Claude 3.5 Sonnet, Opus, Haiku
                - **OpenAI:** GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
                - **Google:** Gemini Pro, Gemini 2.0 Flash
                - **Meta:** Llama 3.1 70B, 405B
                - **Specialized:** DeepSeek Coder, Codestral
                
                **💰 Cost & Usage Tracking:**
                - Real-time token usage monitoring
                - Cost estimation across different models
                - API call tracking and performance metrics
                """)
    
    # Detailed issues section
    if st.session_state.analysis_results and st.session_state.analysis_results.issues:
        st.header("🔍 Detailed Issues Analysis")
        
        issues = st.session_state.analysis_results.issues
        
        # Filtering options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                [s.value for s in Severity],
                default=[s.value for s in Severity],
                key="severity_filter"
            )
        
        with col_filter2:
            category_filter = st.multiselect(
                "Filter by Category",
                [c.value for c in IssueCategory],
                default=[c.value for c in IssueCategory],
                key="category_filter"
            )
        
        with col_filter3:
            sort_by = st.selectbox(
                "Sort by",
                ["Severity", "Confidence", "Line Number", "Category"],
                key="sort_by"
            )
        
        # Apply filters
        filtered_issues = [
            issue for issue in issues
            if (issue.severity.value in severity_filter and 
                issue.category.value in category_filter)
        ]
        
        # Sort issues
        if sort_by == "Severity":
            severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
            filtered_issues.sort(key=lambda x: severity_order.get(x.severity, 4))
        elif sort_by == "Confidence":
            filtered_issues.sort(key=lambda x: x.confidence, reverse=True)
        elif sort_by == "Line Number":
            filtered_issues.sort(key=lambda x: x.line_number)
        else:  # Category
            filtered_issues.sort(key=lambda x: x.category.value)
        
        st.write(f"**Showing {len(filtered_issues)} of {len(issues)} issues**")
        
        # Display issues
        for i, issue in enumerate(filtered_issues):
            severity_emoji = {
                Severity.CRITICAL: "🔴",
                Severity.HIGH: "🟠",
                Severity.MEDIUM: "🟡",
                Severity.LOW: "🔵"
            }[issue.severity]
            
            # Create expandable issue card
            with st.expander(
                f"{severity_emoji} **{issue.title}** | Line {issue.line_number} | {issue.category.value} | Confidence: {issue.confidence:.1f}",
                expanded=(i < 3 and issue.severity in [Severity.CRITICAL, Severity.HIGH])
            ):
                col_issue1, col_issue2 = st.columns([3, 1])
                
                with col_issue1:
                    st.markdown("**📝 Description:**")
                    st.write(issue.description)
                    
                    st.markdown("**💡 Suggested Fix:**")
                    st.write(issue.suggestion)
                    
                    if issue.code_snippet:
                        st.markdown("**📄 Code Snippet:**")
                        st.code(issue.code_snippet, language=language.lower())
                
                with col_issue2:
                    st.markdown("**📊 Issue Details:**")
                    st.write(f"**Severity:** {issue.severity.value}")
                    st.write(f"**Category:** {issue.category.value}")
                    st.write(f"**Line:** {issue.line_number}")
                    st.write(f"**Confidence:** {issue.confidence:.1f}")
                    
                    # Priority indicator
                    if issue.severity == Severity.CRITICAL:
                        st.error("🚨 **CRITICAL** - Fix immediately")
                    elif issue.severity == Severity.HIGH:
                        st.warning("⚠️ **HIGH** - Address soon")
                    elif issue.severity == Severity.MEDIUM:
                        st.info("📋 **MEDIUM** - Plan to fix")
                    else:
                        st.success("✓ **LOW** - Optional improvement")
    
    # Export and sharing section
    if st.session_state.analysis_results:
        st.header("📤 Export & Share Results")
        
        col_export1, col_export2, col_export3, col_export4 = st.columns(4)
        
        with col_export1:
            # JSON export
            if st.button("📋 Export JSON"):
                export_data = {
                    "analysis_result": st.session_state.analysis_results.dict(),
                    "model_info": {
                        "model_used": reviewer.current_model,
                        "provider": "OpenRouter",
                        "analysis_time": st.session_state.analysis_results.analysis_time
                    },
                    "code_info": st.session_state.get('file_info', {}),
                    "langchain_info": {
                        "framework": "LangChain",
                        "parsers_used": ["JsonOutputParser", "PydanticOutputParser"],
                        "chains_used": ["analysis_chain", "summary_chain"]
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                st.download_button(
                    "📥 Download Full Report",
                    json.dumps(export_data, indent=2, default=str),
                    file_name=f"langchain_openrouter_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Complete analysis results with metadata"
                )
        
        with col_export2:
            # Summary export
            if st.button("📝 Export Summary"):
                summary_text = f"""# Code Review Summary
Generated by: {st.session_state.analysis_results.model_used}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Framework: LangChain + OpenRouter

## Quality Score
{st.session_state.analysis_results.quality_score}/100

## Issues Found
- Total: {len(st.session_state.analysis_results.issues)}
- Critical: {len([i for i in st.session_state.analysis_results.issues if i.severity == Severity.CRITICAL])}
- High: {len([i for i in st.session_state.analysis_results.issues if i.severity == Severity.HIGH])}
- Medium: {len([i for i in st.session_state.analysis_results.issues if i.severity == Severity.MEDIUM])}
- Low: {len([i for i in st.session_state.analysis_results.issues if i.severity == Severity.LOW])}

## Executive Summary
{st.session_state.analysis_results.summary}

## Key Recommendations
{chr(10).join(f"- {rec}" for rec in st.session_state.analysis_results.recommendations)}

## Usage Statistics
- Analysis Time: {st.session_state.analysis_results.analysis_time:.1f}s
- Tokens Used: {st.session_state.analysis_results.token_usage.get('total_tokens', 'N/A')}
- Estimated Cost: ${st.session_state.analysis_results.token_usage.get('estimated_cost', 0):.4f}
"""
                
                st.download_button(
                    "📥 Download Summary",
                    summary_text,
                    file_name=f"code_review_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col_export3:
            # Usage stats
            if st.button("📊 Usage Stats"):
                st.json({
                    "model_calls": len(reviewer.callback.model_calls),
                    "total_tokens": reviewer.callback.total_tokens,
                    "estimated_cost": round(reviewer.callback.total_cost, 4),
                    "current_model": reviewer.current_model,
                    "analysis_time": st.session_state.analysis_results.analysis_time,
                    "issues_found": len(st.session_state.analysis_results.issues)
                })
        
        with col_export4:
            # Share link (placeholder)
            if st.button("🔗 Share Results"):
                st.info("🚧 Share functionality coming soon!")
                st.write("For now, use the export options to share your analysis results.")

# Advanced features demonstration
def show_advanced_langchain_patterns():
    """Demonstrate advanced LangChain patterns"""
    st.header("🚀 Advanced LangChain Patterns")
    
    with st.expander("🔗 Chain Composition Examples"):
        st.code("""
# LangChain Expression Language (LCEL) examples

# Basic chain composition
analysis_chain = prompt | llm | json_parser

# Parallel processing multiple aspects
parallel_analysis = RunnableParallel({
    "security": security_prompt | llm | json_parser,
    "performance": performance_prompt | llm | json_parser,
    "maintainability": maintainability_prompt | llm | json_parser
})

# Conditional routing based on code complexity
def route_by_complexity(input_data):
    lines = len(input_data["code"].split('\n'))
    if lines > 1000:
        return detailed_analysis_chain
    else:
        return quick_analysis_chain

routing_chain = RunnableLambda(route_by_complexity)

# Sequential chain with memory
chain_with_context = (
    RunnablePassthrough.assign(
        history=lambda x: memory.chat_memory.messages
    )
    | contextualized_prompt
    | llm
    | json_parser
)
        """, language="python")
    
    with st.expander("🎯 Structured Output with Pydantic"):
        st.code("""
# Type-safe output models
class SecurityVulnerability(BaseModel):
    cve_id: Optional[str] = None
    severity: Literal["Critical", "High", "Medium", "Low"]
    vulnerability_type: str
    affected_lines: List[int]
    exploitability_score: float = Field(ge=0.0, le=10.0)
    remediation_steps: List[str]
    references: List[str] = Field(default_factory=list)

class PerformanceIssue(BaseModel):
    bottleneck_type: str
    current_complexity: str  # e.g., "O(n²)"
    optimized_complexity: str  # e.g., "O(n log n)"
    performance_impact: Literal["High", "Medium", "Low"]
    optimization_suggestion: str
    code_example: str

# Parser integration
security_parser = PydanticOutputParser(pydantic_object=SecurityVulnerability)
performance_parser = PydanticOutputParser(pydantic_object=PerformanceIssue)

# Chain with structured output
security_chain = (
    security_prompt.partial(
        format_instructions=security_parser.get_format_instructions()
    )
    | llm
    | security_parser
)
        """, language="python")
    
    with st.expander("📊 Custom Callbacks and Monitoring"):
        st.code("""
class ComprehensiveCallback(BaseCallbackHandler):
    def __init__(self):
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "token_usage": {},
            "model_switches": [],
            "error_count": 0,
            "chain_executions": []
        }
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.metrics["start_time"] = datetime.now()
        self.metrics["chain_executions"].append({
            "chain": serialized.get("id", ["unknown"])[-1],
            "start_time": datetime.now(),
            "inputs_size": len(str(inputs))
        })
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        model_name = serialized.get("id", ["unknown"])[-1]
        self.metrics["model_switches"].append({
            "model": model_name,
            "timestamp": datetime.now(),
            "prompt_tokens": sum(len(p) for p in prompts)
        })
    
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output'):
            usage = response.llm_output.get('token_usage', {})
            for key, value in usage.items():
                if key in self.metrics["token_usage"]:
                    self.metrics["token_usage"][key] += value
                else:
                    self.metrics["token_usage"][key] = value
    
    def on_chain_error(self, error, **kwargs):
        self.metrics["error_count"] += 1
        logger.error(f"Chain error: {error}")
    
    def get_performance_report(self):
        total_time = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
        return {
            "total_execution_time": total_time,
            "chains_executed": len(self.metrics["chain_executions"]),
            "models_used": len(set(m["model"] for m in self.metrics["model_switches"])),
            "total_tokens": self.metrics["token_usage"].get("total_tokens", 0),
            "errors_encountered": self.metrics["error_count"]
        }

# Usage
callback = ComprehensiveCallback()
chain = prompt | llm.with_config(callbacks=[callback]) | parser
result = chain.invoke({"code": code_content})
performance_report = callback.get_performance_report()
        """, language="python")

if __name__ == "__main__":
    main()
    
    # Advanced features toggle
    with st.sidebar:
        st.divider()
        if st.checkbox("🚀 Show Advanced Patterns", help="Display advanced LangChain usage examples"):
            show_advanced_langchain_patterns()