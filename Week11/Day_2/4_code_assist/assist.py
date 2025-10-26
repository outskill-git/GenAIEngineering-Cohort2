import streamlit as st
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
from datetime import datetime
import asyncio
import concurrent.futures
from functools import lru_cache
import time

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# Tool imports
from pydantic import BaseModel, Field
import requests
import subprocess
import tempfile

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Global configuration for performance
FAST_MODEL = "openai/gpt-3.5-turbo"
ADVANCED_MODEL = "openai/gpt-4"
ULTRA_FAST_MODE = True  # Bypass CrewAI for simple requests

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'


@dataclass
class CodeOutput:
    """Data class to store generated code and documentation."""
    main_code: str
    test_code: str
    documentation: str
    requirements: List[str]
    architecture_notes: str
    generation_time: float = 0.0


class FastCodeAnalysisTool(BaseTool):
    """Ultra-fast code analysis tool."""
    name: str = "fast_code_analysis_tool"
    description: str = "Analyzes code quickly"

    def _run(self, code: str) -> str:
        try:
            compile(code, '<string>', 'exec')
            issues = []
            if 'import *' in code:
                issues.append("Avoid wildcard imports")
            return f"✅ Code validated" if not issues else f"⚠️ {len(issues)} issues found"
        except SyntaxError as e:
            return f"❌ Syntax error: {str(e)}"
        except Exception:
            return "❌ Analysis failed"


class CodeExplanationTool(BaseTool):
    """Tool for explaining code structure."""
    name: str = "code_explanation_tool"
    description: str = "Explains code structure and functionality"

    def _run(self, code: str) -> str:
        try:
            analysis = {
                'functions': re.findall(r'def\s+(\w+)', code),
                'classes': re.findall(r'class\s+(\w+)', code),
                'imports': re.findall(r'(?:from\s+\S+\s+)?import\s+(.+)', code),
                'lines_of_code': len(code.split('\n')),
                'complexity_score': len(re.findall(r'\b(?:if|for|while|try|except)\b', code))
            }
            return json.dumps(analysis, indent=2)
        except Exception as e:
            return f"Analysis error: {str(e)}"


def call_openrouter_direct(prompt: str, model: str = FAST_MODEL, max_tokens: int = 2000) -> str:
    """Direct OpenRouter API call for maximum speed."""
    try:
        headers = {
            'Authorization': f'Bearer {os.getenv("OPEN_ROUTER_KEY")}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://streamlit.io',
            'X-Title': 'Fast Code Generator'
        }
        
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': 0.3,
            'stream': False
        }
        
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"


def generate_code_ultra_fast(requirements: str) -> CodeOutput:
    """Ultra-fast code generation using direct API calls."""
    start_time = time.time()
    
    try:
        # Single comprehensive prompt for all components
        prompt = f"""Generate a complete Python solution for: {requirements}

Please provide your response in the following format:

=== MAIN CODE ===
[Write clean, working Python code with comments]

=== TESTS ===
[Write basic unittest code]

=== DOCUMENTATION ===
[Write a simple README in markdown]

=== REQUIREMENTS ===
[List Python packages needed, one per line]

Keep it simple, functional, and well-commented. Focus on working code over complex features."""

        # Get response
        response = call_openrouter_direct(prompt, ADVANCED_MODEL, 3000)
        
        # Parse response sections
        main_code = ""
        test_code = ""
        documentation = ""
        requirements_text = ""
        
        sections = {
            'MAIN CODE': '',
            'TESTS': '',
            'DOCUMENTATION': '',
            'REQUIREMENTS': ''
        }
        
        current_section = None
        lines = response.split('\n')
        
        for line in lines:
            line_upper = line.upper().strip()
            if '===' in line_upper and any(section in line_upper for section in sections.keys()):
                for section in sections.keys():
                    if section in line_upper:
                        current_section = section
                        break
            elif current_section:
                sections[current_section] += line + '\n'
        
        main_code = sections['MAIN CODE'].strip()
        test_code = sections['TESTS'].strip()
        documentation = sections['DOCUMENTATION'].strip()
        requirements_text = sections['REQUIREMENTS'].strip()
        
        # Parse requirements
        requirements = []
        if requirements_text:
            requirements = [req.strip() for req in requirements_text.split('\n') if req.strip() and not req.startswith('#')]
        
        # Clean up code sections
        main_code = clean_code_block(main_code)
        test_code = clean_code_block(test_code)
        
        generation_time = time.time() - start_time
        
        return CodeOutput(
            main_code=main_code,
            test_code=test_code,
            documentation=documentation,
            requirements=requirements,
            architecture_notes=f"Generated using ultra-fast mode in {generation_time:.1f}s",
            generation_time=generation_time
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        return CodeOutput(
            main_code=f"# Error generating code: {str(e)}",
            test_code="# Test generation failed",
            documentation=f"# Generation failed: {str(e)}",
            requirements=[],
            architecture_notes=f"Failed in {generation_time:.1f}s",
            generation_time=generation_time
        )


def clean_code_block(code_text: str) -> str:
    """Clean code blocks from markdown formatting."""
    # Remove markdown code blocks
    code_text = re.sub(r'```python\n?', '', code_text)
    code_text = re.sub(r'```\n?', '', code_text)
    
    # Remove extra whitespace
    lines = code_text.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)


def explain_code_fast(code: str) -> Dict[str, str]:
    """Fast code explanation using direct API call."""
    start_time = time.time()
    
    try:
        prompt = f"""Analyze and explain this Python code in detail:

```python
{code}
```

Provide a comprehensive explanation including:
1. **Overview**: What does this code do?
2. **Structure**: Classes, functions, and their purposes
3. **Logic Flow**: How the code executes
4. **Dependencies**: External libraries used
5. **Best Practices**: Code quality assessment
6. **Improvements**: Suggestions for enhancement
7. **Usage**: How to use this code

Be educational and detailed in your explanation."""

        explanation = call_openrouter_direct(prompt, ADVANCED_MODEL, 2000)
        
        # Also get structured analysis
        analysis_tool = CodeExplanationTool()
        structure_analysis = analysis_tool._run(code)
        
        # Calculate complexity
        complexity_score = len(re.findall(r'\b(?:if|for|while|try|except|and|or)\b', code))
        complexity_level = "Low" if complexity_score < 10 else "Medium" if complexity_score < 25 else "High"
        
        analysis_time = time.time() - start_time
        
        return {
            'explanation': explanation,
            'structure_analysis': structure_analysis,
            'complexity_analysis': json.dumps({
                'complexity_score': complexity_score,
                'complexity_level': complexity_level,
                'lines_of_code': len(code.split('\n')),
                'function_count': len(re.findall(r'def\s+\w+', code)),
                'class_count': len(re.findall(r'class\s+\w+', code))
            }, indent=2),
            'code_length': len(code.split('\n')),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_time': analysis_time
        }
        
    except Exception as e:
        return {
            'explanation': f"Analysis failed: {str(e)}",
            'structure_analysis': "{}",
            'complexity_analysis': "{}",
            'code_length': len(code.split('\n')),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_time': time.time() - start_time
        }


@lru_cache(maxsize=32)
def validate_api_key_cached(api_key: str) -> bool:
    """Cached API key validation."""
    if not api_key or not api_key.startswith('sk-'):
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.get(
            'https://openrouter.ai/api/v1/models',
            headers=headers,
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def setup_llm(model: str, api_key: str) -> LLM:
    """Setup LLM for CrewAI (fallback mode)."""
    return LLM(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3
    )


def run_crewai_fallback(user_requirements: str, api_key: str) -> CodeOutput:
    """Fallback to CrewAI for complex requests."""
    start_time = time.time()
    
    try:
        # Create minimal agents
        llm = setup_llm(FAST_MODEL, api_key)
        
        developer = Agent(
            role='Developer',
            goal='Write working Python code',
            backstory='You write clean, functional Python code quickly.',
            verbose=False,
            allow_delegation=False,
            llm=llm,
            max_iter=1,
            max_execution_time=60
        )

        # Single comprehensive task
        dev_task = Task(
            description=f"""Create a complete Python solution for: {user_requirements}
            
            Include:
            1. Working Python code with comments
            2. Basic unit tests
            3. Simple documentation
            4. List of requirements
            
            Focus on functionality over perfection.""",
            expected_output="Complete Python solution with code, tests, and docs",
            agent=developer
        )

        # Run crew
        crew = Crew(
            agents=[developer],
            tasks=[dev_task],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        
        # Fixed output parsing for newer CrewAI versions
        try:
            if hasattr(dev_task.output, 'raw'):
                output_text = dev_task.output.raw
            elif hasattr(dev_task.output, 'result'):
                output_text = dev_task.output.result
            elif hasattr(dev_task.output, 'content'):
                output_text = dev_task.output.content
            else:
                output_text = str(dev_task.output)
        except:
            output_text = str(result)

        # Simple parsing of the output
        main_code = extract_section(output_text, ['```python', 'def ', 'class ', 'import '])
        test_code = extract_section(output_text, ['import unittest', 'def test_'])
        documentation = extract_section(output_text, ['# ', '## ', '### '])
        
        generation_time = time.time() - start_time
        
        return CodeOutput(
            main_code=main_code or "# Code generation incomplete",
            test_code=test_code or "# Tests not generated",
            documentation=documentation or "# Documentation not generated",
            requirements=['requests'],  # Default requirement
            architecture_notes=f"Generated using CrewAI fallback in {generation_time:.1f}s",
            generation_time=generation_time
        )

    except Exception as e:
        generation_time = time.time() - start_time
        raise Exception(f"CrewAI fallback failed in {generation_time:.1f}s: {str(e)}")


def extract_section(text: str, indicators: List[str]) -> str:
    """Extract a section of text based on indicators."""
    for indicator in indicators:
        start = text.find(indicator)
        if start != -1:
            # Find a reasonable end point
            end = start + 1000  # Limit section size
            section = text[start:end]
            return clean_code_block(section)
    return ""


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="⚡ Ultra-Fast Code Generator",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ Ultra-Fast AI Code Generator & Explainer")
    st.markdown("""
    **Lightning Fast Code Generation** ⚡ Optimized for speed and simplicity!
    
    🚀 **Ultra-Fast Mode**: Direct API calls (5-15 seconds)  
    🔍 **Code Explainer**: Understand any Python code instantly  
    🛠️ **Fallback Mode**: CrewAI for complex requests  
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = os.getenv("OPEN_ROUTER_KEY")
        
        # Mode selection
        app_mode = st.selectbox(
            "Choose Mode",
            ["🚀 Code Generator", "🔍 Code Explainer"],
            help="Generate new code or explain existing code"
        )
        
        if app_mode == "🚀 Code Generator":
            # Generation options
            use_ultra_fast = st.checkbox(
                "⚡ Ultra-Fast Mode", 
                value=True,
                help="Direct API calls for maximum speed (recommended)"
            )
            
            if not use_ultra_fast:
                st.warning("CrewAI mode is slower but may handle complex requests better")
        
        # Performance metrics
        if 'last_generation_time' in st.session_state:
            st.metric(
                "Last Generation Time", 
                f"{st.session_state.last_generation_time:.1f}s"
            )
        
        # Quick examples
        st.subheader("🚀 Quick Examples")
        quick_examples = [
            "Simple calculator with GUI",
            "File organizer by extension",
            "Password generator with options",
            "Todo list with save/load",
            "Basic web scraper",
            "CSV data analyzer",
            "Simple REST API with Flask"
        ]
        
        for example in quick_examples:
            if st.button(f"📝 {example}", key=f"quick_{example}"):
                st.session_state.quick_requirements = example

    # Main content based on mode
    if app_mode == "🚀 Code Generator":
        # Code Generation Mode
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📝 Requirements")
            
            # Use quick example if selected
            default_text = st.session_state.get('quick_requirements', '')
            
            user_requirements = st.text_area(
                "What do you want to build?",
                value=default_text,
                height=120,
                placeholder="Example: Create a simple calculator with add, subtract, multiply, divide functions and a basic GUI"
            )
            
            # Generate button
            generate_label = "⚡ Generate Ultra-Fast" if use_ultra_fast else "🚀 Generate with CrewAI"
            
            if st.button(generate_label, type="primary", disabled=not (api_key and user_requirements)):
                if not validate_api_key_cached(api_key):
                    st.error("❌ Invalid API key")
                    st.stop()

                with st.spinner(f"{'⚡ Generating at lightning speed...' if use_ultra_fast else '🤖 CrewAI agents working...'}"):
                    progress_bar = st.progress(0)
                    
                    try:
                        progress_bar.progress(25)
                        
                        if use_ultra_fast:
                            result = generate_code_ultra_fast(user_requirements)
                        else:
                            result = run_crewai_fallback(user_requirements, api_key)
                        
                        progress_bar.progress(100)
                        
                        # Store results
                        st.session_state.code_result = result
                        st.session_state.generation_complete = True
                        st.session_state.last_generation_time = result.generation_time
                        
                        st.success(f"✅ Generated in {result.generation_time:.1f} seconds!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        progress_bar.empty()

        with col2:
            st.header("📊 Status")
            
            if hasattr(st.session_state, 'generation_complete') and st.session_state.generation_complete:
                result = st.session_state.code_result
                
                # Performance metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("⚡ Time", f"{result.generation_time:.1f}s")
                    st.metric("📝 Lines", len(result.main_code.split('\n')))
                with col_b:
                    st.metric("📦 Deps", len(result.requirements))
                    st.metric("🧪 Tests", result.test_code.count('def test_'))
                    
                # Speed indicator
                if result.generation_time < 15:
                    st.success("🚀 Lightning Fast!")
                elif result.generation_time < 30:
                    st.info("⚡ Fast")
                else:
                    st.warning("🐌 Slow")
                    
            else:
                st.info("👆 Enter requirements and click generate")

    elif app_mode == "🔍 Code Explainer":
        # Code Explanation Mode
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("📝 Code Input")

            # Sample code options
            sample_codes = {
                "": "",
                "Simple Function": """def fibonacci(n):
    '''Calculate nth Fibonacci number.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")""",

                "Class Example": """class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance
        self.transactions = []

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            self.transactions.append(f"Deposit: +${amount}")
            return True
        return False

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            self.transactions.append(f"Withdrawal: -${amount}")
            return True
        return False""",

                "Web Scraper": """import requests
from bs4 import BeautifulSoup
import json

def scrape_headlines(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        headlines = []
        for h in soup.find_all(['h1', 'h2', 'h3']):
            if h.text.strip():
                headlines.append(h.text.strip())
        
        return headlines[:10]  # Top 10
    except Exception as e:
        return f"Error: {e}"

# Usage
url = "https://example.com"
headlines = scrape_headlines(url)
print(json.dumps(headlines, indent=2))"""
            }

            selected_sample = st.selectbox(
                "Choose sample code:",
                list(sample_codes.keys())
            )

            code_to_explain = st.text_area(
                "Paste your Python code here:",
                value=sample_codes[selected_sample],
                height=300,
                placeholder="Paste any Python code you want to understand..."
            )

            # File upload
            uploaded_file = st.file_uploader(
                "Or upload a Python file:",
                type=['py']
            )

            if uploaded_file:
                code_to_explain = str(uploaded_file.read(), "utf-8")
                st.success(f"📁 Loaded {uploaded_file.name}")

            # Explain button
            if st.button("🔍 Explain Code", type="primary", disabled=not (api_key and code_to_explain.strip())):
                if not validate_api_key_cached(api_key):
                    st.error("❌ Invalid API key")
                    st.stop()

                with st.spinner("🔍 Analyzing your code..."):
                    progress_bar = st.progress(0)
                    
                    try:
                        progress_bar.progress(50)
                        
                        explanation_result = explain_code_fast(code_to_explain)
                        
                        progress_bar.progress(100)
                        
                        st.session_state.explanation_result = explanation_result
                        st.session_state.explanation_complete = True
                        
                        st.success(f"✅ Analysis completed in {explanation_result['analysis_time']:.1f}s!")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        progress_bar.empty()

        with col2:
            st.header("📊 Analysis Status")

            if hasattr(st.session_state, 'explanation_complete') and st.session_state.explanation_complete:
                result = st.session_state.explanation_result

                # Analysis metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("⚡ Time", f"{result['analysis_time']:.1f}s")
                    st.metric("📝 Lines", result['code_length'])
                with col_b:
                    try:
                        struct_data = json.loads(result['structure_analysis'])
                        st.metric("⚙️ Functions", len(struct_data.get('functions', [])))
                        st.metric("🏛️ Classes", len(struct_data.get('classes', [])))
                    except:
                        st.metric("⚙️ Functions", "N/A")
                        st.metric("🏛️ Classes", "N/A")

                # Complexity indicator
                try:
                    complexity_data = json.loads(result['complexity_analysis'])
                    complexity_level = complexity_data.get('complexity_level', 'Unknown')
                    if complexity_level == 'Low':
                        st.success("🟢 Low Complexity")
                    elif complexity_level == 'Medium':
                        st.info("🟡 Medium Complexity")
                    else:
                        st.warning("🔴 High Complexity")
                except:
                    st.info("❓ Complexity Unknown")
                    
            else:
                st.info("👆 Input code and click analyze")

    # Results display for Code Generation
    if app_mode == "🚀 Code Generator" and hasattr(st.session_state, 'code_result'):
        result = st.session_state.code_result

        st.header("📋 Generated Code")

        tab1, tab2, tab3, tab4 = st.tabs(["💻 Main Code", "🧪 Tests", "📚 Documentation", "📦 Setup"])

        with tab1:
            st.subheader("Main Implementation")
            if result.main_code:
                st.code(result.main_code, language="python")
                st.download_button(
                    "⬇️ Download main.py",
                    result.main_code,
                    "main.py",
                    "text/plain"
                )
            else:
                st.warning("No main code generated")

        with tab2:
            st.subheader("Test Suite")
            if result.test_code:
                st.code(result.test_code, language="python")
                st.download_button(
                    "⬇️ Download tests.py",
                    result.test_code,
                    "tests.py",
                    "text/plain"
                )
            else:
                st.info("No tests generated")

        with tab3:
            st.subheader("Documentation")
            if result.documentation:
                st.markdown(result.documentation)
                st.download_button(
                    "⬇️ Download README.md",
                    result.documentation,
                    "README.md",
                    "text/plain"
                )
            else:
                st.info("No documentation generated")

        with tab4:
            st.subheader("Requirements & Setup")
            if result.requirements:
                requirements_text = '\n'.join(result.requirements)
                st.code(requirements_text, language="text")
                st.download_button(
                    "⬇️ Download requirements.txt",
                    requirements_text,
                    "requirements.txt",
                    "text/plain"
                )
            else:
                st.info("No external dependencies")
                
            if result.architecture_notes:
                st.subheader("🏗️ Architecture Notes")
                st.text(result.architecture_notes)

    # Results display for Code Explanation
    elif app_mode == "🔍 Code Explainer" and hasattr(st.session_state, 'explanation_result'):
        result = st.session_state.explanation_result

        st.header("🔍 Code Analysis Results")

        tab1, tab2, tab3, tab4 = st.tabs(["📖 Explanation", "🏗️ Structure", "📊 Complexity", "💾 Export"])

        with tab1:
            st.subheader("Detailed Code Explanation")
            st.markdown(result['explanation'])

        with tab2:
            st.subheader("Code Structure")
            try:
                structure_data = json.loads(result['structure_analysis'])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if structure_data.get('functions'):
                        st.write("**⚙️ Functions:**")
                        for func in structure_data['functions']:
                            st.write(f"- `{func}`")
                    
                    if structure_data.get('classes'):
                        st.write("**🏛️ Classes:**")
                        for cls in structure_data['classes']:
                            st.write(f"- `{cls}`")
                
                with col_b:
                    if structure_data.get('imports'):
                        st.write("**📦 Imports:**")
                        for imp in structure_data['imports'][:5]:  # Show first 5
                            st.write(f"- `{imp.strip()}`")
                
                st.json(structure_data)
                
            except:
                st.code(result['structure_analysis'])

        with tab3:
            st.subheader("Complexity Analysis")
            try:
                complexity_data = json.loads(result['complexity_analysis'])
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Complexity Score", complexity_data.get('complexity_score', 'N/A'))
                with col_b:
                    st.metric("Complexity Level", complexity_data.get('complexity_level', 'N/A'))
                with col_c:
                    st.metric("Functions", complexity_data.get('function_count', 'N/A'))
                
                st.json(complexity_data)
                
            except:
                st.code(result['complexity_analysis'])

        with tab4:
            st.subheader("Export Analysis")
            
            # Create full report
            full_report = f"""# Code Analysis Report

**Analysis Date:** {result['analysis_timestamp']}  
**Analysis Time:** {result['analysis_time']:.1f} seconds  
**Code Length:** {result['code_length']} lines  

## Detailed Explanation
{result['explanation']}

## Structure Analysis
```json
{result['structure_analysis']}
```

## Complexity Analysis
```json
{result['complexity_analysis']}
```
"""
            
            st.download_button(
                "⬇️ Download Full Report",
                full_report,
                f"code_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                "text/plain"
            )


if __name__ == "__main__":
    main()