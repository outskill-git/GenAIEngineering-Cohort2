import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from tavily import TavilyClient
import wikipedia
import arxiv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import tempfile
import shutil
import json

# ----------------------------
# Streamlit App Configuration
# ----------------------------
st.set_page_config(page_title="Agentic RAG with Tavily + OpenRouter", layout="wide")
st.title("🔍 Agentic RAG with Tavily + GPT-4 (OpenRouter) + HuggingFace Embeddings")

# ----------------------------
# API Key Inputs
# ----------------------------
st.sidebar.header("API Keys")
openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password")
tavily_key = st.sidebar.text_input("Tavily API Key", type="password")

# ----------------------------
# Query Examples
# ----------------------------
st.sidebar.header("📝 Query Examples")
st.sidebar.markdown("""
**The system will automatically choose the best source:**

- "Who is Einstein?" → Wikipedia  
- "Latest research on quantum computing" → ArXiv  
- "Today's weather" → Web Search  
- Questions about uploaded PDFs → RAG first, then fallback

**Available Tools:**
- 📄 PDF RAG (your documents)
- 📚 Wikipedia (encyclopedic info)
- 🔬 ArXiv (academic papers)
- 🌐 Web Search (current info)
""")

# HuggingFace model for embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Global variable to track source
LAST_SOURCE_INFO = {
    "source": None,
    "urls": [],
    "fallback_from": None
}

retriever = None

if openrouter_key and tavily_key:
    try:
        # Set API Keys
        os.environ["OPENAI_API_KEY"] = openrouter_key
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        os.environ["TAVILY_API_KEY"] = tavily_key

        # ----------------------------
        # Initialize Tavily + LLM
        # ----------------------------
        tavily_client = TavilyClient(api_key=tavily_key)

        def tavily_search(query: str) -> str:
            """Search the web using Tavily and return formatted results with source tracking"""
            try:
                response = tavily_client.search(query)
                results = response.get("results", [])
                
                if not results:
                    return "NO_RESULTS: No search results found."
                
                formatted_results = []
                sources_list = []
                urls = []
                
                for i, result in enumerate(results[:5], 1):
                    title = result.get("title", "No title")
                    content = result.get("content", "No content")
                    url = result.get("url", "No URL")
                    formatted_results.append(f"{i}. {title}\n{content}\n")
                    sources_list.append(f"- [{title}]({url})")
                    urls.append(url)
                
                # Store source info globally
                LAST_SOURCE_INFO["urls"] = urls
                
                result_text = "\n".join(formatted_results)
                sources_text = "\n".join(sources_list)
                
                return f"{result_text}\n\n**Web Sources:**\n{sources_text}"
                
            except Exception as e:
                return f"ERROR: Failed to search web - {str(e)}"

        def wikipedia_search(query: str) -> str:
            """Search Wikipedia for encyclopedic information with fallback"""
            try:
                # Set language to English and limit results
                wikipedia.set_lang("en")
                
                # Search for articles
                search_results = wikipedia.search(query, results=3)
                
                if not search_results:
                    # Fallback: Try with simplified query (first few words)
                    simplified_query = ' '.join(query.split()[:3])
                    search_results = wikipedia.search(simplified_query, results=3)
                    
                    if not search_results:
                        return "NO_RESULTS: No Wikipedia articles found."
                
                # Try to get the first article
                for article_name in search_results:
                    try:
                        page = wikipedia.page(article_name)
                        summary = wikipedia.summary(article_name, sentences=4)
                        
                        # Store source info globally
                        LAST_SOURCE_INFO["urls"] = [page.url]
                        
                        result = f"**Wikipedia Article: {page.title}**\n\n"
                        result += f"{summary}\n\n"
                        
                        # Add alternative articles if available
                        if len(search_results) > 1:
                            other_articles = [a for a in search_results if a != article_name]
                            result += f"Related articles: {', '.join(other_articles[:2])}\n"
                        
                        result += f"\n**Wikipedia Source:** [{page.title}]({page.url})"
                        return result
                        
                    except wikipedia.exceptions.DisambiguationError as e:
                        # Handle disambiguation pages - try first option
                        if e.options:
                            try:
                                page = wikipedia.page(e.options[0])
                                summary = wikipedia.summary(e.options[0], sentences=4)
                                
                                # Store source info globally
                                LAST_SOURCE_INFO["urls"] = [page.url]
                                
                                result = f"**Wikipedia Article: {page.title}** (from disambiguation)\n\n"
                                result += f"{summary}\n\n"
                                result += f"Other options: {', '.join(e.options[1:5])}\n"
                                result += f"\n**Wikipedia Source:** [{page.title}]({page.url})"
                                return result
                            except:
                                continue
                                
                    except wikipedia.exceptions.PageError:
                        continue
                
                return "NO_RESULTS: Wikipedia pages not accessible for this query."
                    
            except Exception as e:
                return f"ERROR: Wikipedia search failed - {str(e)}"

        def arxiv_search(query: str) -> str:
            """Search ArXiv for academic papers with improved fallback"""
            try:
                # Create search client
                client = arxiv.Client()
                
                # First attempt with original query
                search = arxiv.Search(
                    query=query,
                    max_results=5,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                results_list = list(client.results(search))
                
                # If no results, try with broader search terms
                if not results_list:
                    # Remove special characters and try again
                    simplified_query = ' '.join(query.replace('"', '').replace("'", '').split()[:5])
                    search = arxiv.Search(
                        query=simplified_query,
                        max_results=5,
                        sort_by=arxiv.SortCriterion.Relevance
                    )
                    results_list = list(client.results(search))
                
                if not results_list:
                    return "NO_RESULTS: No academic papers found on ArXiv."
                
                formatted_results = []
                sources_list = []
                urls = []
                
                for i, result in enumerate(results_list, 1):
                    paper_info = f"**{i}. {result.title}**\n"
                    paper_info += f"Authors: {', '.join([author.name for author in result.authors[:3]])}\n"
                    paper_info += f"Published: {result.published.strftime('%Y-%m-%d')}\n"
                    paper_info += f"Abstract: {result.summary[:300]}...\n"
                    formatted_results.append(paper_info)
                    
                    sources_list.append(f"- [{result.title}]({result.entry_id})")
                    urls.append(result.entry_id)
                
                # Store source info globally
                LAST_SOURCE_INFO["urls"] = urls
                
                result_text = "\n".join(formatted_results)
                sources_text = "\n".join(sources_list)
                
                return f"{result_text}\n\n**ArXiv Papers:**\n{sources_text}"
                
            except Exception as e:
                return f"ERROR: ArXiv search failed - {str(e)}"

        def smart_router(query: str) -> str:
            """Intelligently route queries with robust fallback mechanism"""
            global LAST_SOURCE_INFO
            
            # Reset source info
            LAST_SOURCE_INFO = {
                "source": None,
                "urls": [],
                "fallback_from": None
            }
            
            query_lower = query.lower()
            
            # Keywords that strongly suggest ArXiv search
            arxiv_keywords = [
                'research', 'paper', 'study', 'academic', 'scientific', 'publication',
                'latest research', 'recent research', 'research on', 'studies on',
                'academic paper', 'scientific paper', 'peer review', 'journal',
                'findings', 'breakthrough', 'advancement', 'discovery'
            ]
            
            # Keywords that suggest Wikipedia
            wikipedia_keywords = [
                'who is', 'what is', 'define', 'definition', 'biography', 'history of',
                'founded', 'born', 'died', 'invented', 'discovered', 'theory of',
                'concept of', 'principle of', 'explain'
            ]
            
            # Track which sources were tried and their results
            attempts = []
            
            # Determine primary source based on keywords
            primary_source = None
            if any(keyword in query_lower for keyword in arxiv_keywords):
                primary_source = "arxiv"
            elif any(keyword in query_lower for keyword in wikipedia_keywords):
                primary_source = "wikipedia"
            else:
                primary_source = "web"
            
            # Try primary source first
            if primary_source == "arxiv":
                result = arxiv_search(query)
                attempts.append("ArXiv")
                if not result.startswith("NO_RESULTS") and not result.startswith("ERROR"):
                    LAST_SOURCE_INFO["source"] = "ArXiv Academic Papers"
                    return result
                    
            elif primary_source == "wikipedia":
                result = wikipedia_search(query)
                attempts.append("Wikipedia")
                if not result.startswith("NO_RESULTS") and not result.startswith("ERROR"):
                    LAST_SOURCE_INFO["source"] = "Wikipedia Encyclopedia"
                    return result
                    
            elif primary_source == "web":
                result = tavily_search(query)
                attempts.append("Web Search")
                if not result.startswith("NO_RESULTS") and not result.startswith("ERROR"):
                    LAST_SOURCE_INFO["source"] = "Web Search (Tavily)"
                    return result
            
            # Store what we're falling back from
            LAST_SOURCE_INFO["fallback_from"] = primary_source
            
            # Fallback sequence if primary source failed
            fallback_order = []
            if primary_source == "arxiv":
                fallback_order = ["web", "wikipedia"]
            elif primary_source == "wikipedia":
                fallback_order = ["web", "arxiv"]
            else:  # primary was web
                fallback_order = ["wikipedia", "arxiv"]
            
            for fallback_source in fallback_order:
                if fallback_source == "arxiv" and "ArXiv" not in attempts:
                    result = arxiv_search(query)
                    attempts.append("ArXiv")
                    if not result.startswith("NO_RESULTS") and not result.startswith("ERROR"):
                        LAST_SOURCE_INFO["source"] = f"ArXiv Academic Papers (Fallback from {primary_source})"
                        return result
                        
                elif fallback_source == "wikipedia" and "Wikipedia" not in attempts:
                    result = wikipedia_search(query)
                    attempts.append("Wikipedia")
                    if not result.startswith("NO_RESULTS") and not result.startswith("ERROR"):
                        LAST_SOURCE_INFO["source"] = f"Wikipedia Encyclopedia (Fallback from {primary_source})"
                        return result
                        
                elif fallback_source == "web" and "Web Search" not in attempts:
                    result = tavily_search(query)
                    attempts.append("Web Search")
                    if not result.startswith("NO_RESULTS") and not result.startswith("ERROR"):
                        LAST_SOURCE_INFO["source"] = f"Web Search via Tavily (Fallback from {primary_source})"
                        return result
            
            # If all sources failed, return a comprehensive failure message
            LAST_SOURCE_INFO["source"] = "Search Failed"
            return f"Unable to find relevant information from any source.\nAttempted sources: {', '.join(attempts)}\n\nPlease try rephrasing your query or being more specific."

        # Create a single smart routing tool
        smart_tool = Tool(
            name="smart_search",
            func=smart_router,
            description="Intelligently searches across Wikipedia, ArXiv, and web sources based on query type with robust fallback."
        )

        # Initialize LLM (via OpenRouter)
        llm = ChatOpenAI(
            model="openai/gpt-4o", 
            temperature=0,
            openai_api_base="https://openrouter.ai/api/v1"
        )

        # Create agent with the smart routing tool
        tools = [smart_tool]
        
        # Use standard ReAct prompt which handles parsing correctly
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

        # ----------------------------
        # PDF Upload + RAG Setup
        # ----------------------------
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files for RAG context (optional)", 
            type=["pdf"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            try:
                all_docs = []
                temp_dir = tempfile.mkdtemp()
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load and process each PDF
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)

                if all_docs:
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    texts = text_splitter.split_documents(all_docs)

                    # HuggingFace embeddings
                    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                    db = FAISS.from_documents(texts, embeddings)
                    retriever = db.as_retriever(search_kwargs={"k": 5})

                    st.success(f"{len(uploaded_files)} PDF(s) uploaded and indexed for retrieval!")
                
                # Clean up temp files
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

        # ----------------------------
        # Query Input
        # ----------------------------
        query = st.text_input("Enter your query")

        if st.button("Run Query"):
            if query:
                try:
                    answer_found_in_pdf = False
                    
                    # First, try RAG if PDFs are uploaded
                    if retriever:
                        # Try to get an answer from RAG
                        rag_prompt = ChatPromptTemplate.from_template("""
                        Answer the question based on the following context. 
                        If the context doesn't contain enough information to answer the question completely,
                        say "INSUFFICIENT_CONTEXT" at the beginning of your response.

                        Context: {context}

                        Question: {input}
                        
                        Answer:""")
                        
                        document_chain = create_stuff_documents_chain(llm, rag_prompt)
                        retrieval_chain = create_retrieval_chain(retriever, document_chain)
                        
                        rag_result = retrieval_chain.invoke({"input": query})
                        rag_answer = rag_result["answer"].strip()
                        
                        # Check if RAG found relevant information
                        if "INSUFFICIENT_CONTEXT" not in rag_answer and len(rag_answer) > 20:
                            # RAG found relevant info
                            answer_found_in_pdf = True
                            
                            # Create columns for better layout
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.subheader("📄 Answer from Uploaded PDFs")
                            with col2:
                                st.success("**Source: PDF Documents**")
                            
                            st.write(rag_answer)
                            
                            # Show source documents with better formatting
                            if "context" in rag_result and rag_result["context"]:
                                with st.expander("📋 View Source Documents"):
                                    st.markdown("### Documents Used for Answer:")
                                    for i, doc in enumerate(rag_result["context"], start=1):
                                        source_info = f"Page: {doc.metadata.get('page', 'N/A')}"
                                        file_name = doc.metadata.get('source', 'PDF')
                                        if file_name:
                                            file_name = os.path.basename(file_name)
                                        
                                        st.markdown(f"**Source {i}:** `{file_name}` - {source_info}")
                                        st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                        st.markdown("---")
                        else:
                            st.info("📄 No relevant information found in uploaded PDFs. Searching external sources...")
                    
                    # If answer not found in PDF or no PDFs uploaded, use agent for external search
                    if not answer_found_in_pdf:
                        # Use the agent executor which will automatically route through smart_router
                        with st.spinner("Searching external sources..."):
                            agent_response = agent_executor.invoke({"input": query})
                            response_text = agent_response["output"]
                            
                            # Get the source info from global variable
                            source_info = LAST_SOURCE_INFO["source"]
                            urls = LAST_SOURCE_INFO["urls"]
                            fallback_from = LAST_SOURCE_INFO["fallback_from"]
                            
                            # Display with proper source attribution
                            if source_info and source_info != "Search Failed":
                                # Determine icon based on source
                                if "ArXiv" in source_info:
                                    icon = "🔬"
                                    header = "Academic Research Response"
                                elif "Wikipedia" in source_info:
                                    icon = "📚"
                                    header = "Encyclopedia Response"
                                elif "Web Search" in source_info or "Tavily" in source_info:
                                    icon = "🌐"
                                    header = "Web Search Response"
                                else:
                                    icon = "🤖"
                                    header = "Agent Response"
                                
                                # Create columns for better layout
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.subheader(f"{icon} {header}")
                                with col2:
                                    if "Fallback" in source_info:
                                        st.warning(f"**Source:** {source_info}")
                                    else:
                                        st.success(f"**Source:** {source_info}")
                                
                                # Display the response
                                st.write(response_text)
                                
                                # Show source URLs if available
                                if urls:
                                    with st.expander("🔗 View Sources"):
                                        st.markdown("### Information Sources:")
                                        for url in urls:
                                            st.markdown(f"- {url}")
                                            
                            elif source_info == "Search Failed":
                                st.subheader("⚠️ Search Results")
                                st.error("**Source:** All search methods failed")
                                st.write(response_text)
                            else:
                                # Fallback display if source tracking failed
                                st.subheader("🤖 Agent Response")
                                st.warning("**Source:** Unable to determine (tracking error)")
                                st.write(response_text)
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.write("Attempting final fallback to basic web search...")
                    try:
                        basic_search = tavily_search(query)
                        st.subheader("🌐 Basic Web Search Results (Emergency Fallback)")
                        st.warning("**Source:** Direct Tavily API Call (Fallback Mode)")
                        st.write(basic_search)
                    except Exception as e2:
                        st.error(f"All search methods failed: {str(e2)}")
                        st.info("Please try:\n- Rephrasing your query\n- Being more specific\n- Checking your API keys")
            else:
                st.warning("Please enter a query.")

    except Exception as e:
        st.error(f"Initialization error: {str(e)}")

else:
    st.warning("Please provide both OpenRouter and Tavily API keys in the sidebar to continue.")