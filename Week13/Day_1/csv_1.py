# imports
import streamlit as st
import os, tempfile
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import matplotlib.pyplot as plt
import asyncio
import json

# Langsmith integration
import langsmith
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable Langsmith tracing (CORRECT format from documentation)
langsmith_key = os.getenv('LANGSMITH_API_KEY')
if langsmith_key:
    langsmith_client = langsmith.Client(api_key=langsmith_key)
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = langsmith_key
    os.environ["LANGSMITH_PROJECT"] = "CSV App"
    # Set for backward compatibility
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "CSV App"

# Pydantic models for structured responses (moved to module level)
class CSVResponse(BaseModel):
    """Structured response for CSV queries"""
    answer: str = Field(description="The main answer to the user's question")
    confidence: float = Field(description="Confidence score from 0-1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of how the answer was derived")
    data_used: List[str] = Field(description="Key data points used from the CSV", default_factory=list)

class AnalysisResponse(BaseModel):
    """Structured response for CSV analysis"""
    summary: str = Field(description="Summary of the analysis")
    insights: List[str] = Field(description="Key insights discovered", default_factory=list)
    recommendations: List[str] = Field(description="Recommended actions", default_factory=list)

class SummaryResponse(BaseModel):
    """Structured response for CSV summarization"""
    main_summary: str = Field(description="Main summary of the CSV content")
    key_points: List[str] = Field(description="Most important points", default_factory=list)
    data_quality: str = Field(description="Assessment of data quality")

# MUST be the first Streamlit command
st.set_page_config(page_title="CSV AI", layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def home_page():
    st.write("""Select any one feature from above sliderbox: \n
    1. Chat with CSV \n
    2. Analyze CSV  """)

@st.cache_resource()
def get_embeddings_model():
    """Initialize and cache the embedding model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Remove caching from retriever_func to avoid API key issues
def retriever_func(uploaded_file):
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200, 
                        add_start_index=True
                        )
        all_splits = text_splitter.split_documents(data)

        # Use free HuggingFace embeddings (no API key required)
        embeddings = get_embeddings_model()
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        
        # Clean up temp file
        os.remove(tmp_file_path)
        return retriever, vectorstore
    else:
        st.info("Please upload CSV documents to continue.")
        st.stop()

def chat(temperature, model_name, user_api_key):
    st.write("# Talk to CSV")
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")

    if not uploaded_file:
        st.info("Please upload a CSV file to start chatting.")
        return

    # Check if API key is valid
    if not user_api_key or user_api_key == "":
        st.error("❌ Please enter your OpenRouter API key in the sidebar to use this functionality.")
        return

    try:
        retriever, vectorstore = retriever_func(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error processing CSV file: {str(e)}")
        return

    # Configure LLM for OpenRouter with explicit API key
    try:
        # Reduce max_tokens to fit within credit limit
        max_tokens = 1000  # Reduced from default to avoid credit issues

        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True,
            max_tokens=max_tokens,
            base_url="https://openrouter.ai/api/v1",
            api_key=user_api_key,
            default_headers={
                "HTTP-Referer": "https://localhost:8501",
                "X-Title": "CSV AI App"
            }
        )
        if "grok" in model_name.lower():
            st.success(f"🤖 Connected to {model_name} (max_tokens: {max_tokens}) - FREE MODEL! 🎉")
        else:
            st.success(f"🤖 Connected to {model_name} (max_tokens: {max_tokens})")
    except Exception as e:
        error_msg = str(e)
        if "402" in error_msg and "credits" in error_msg:
            st.error("❌ **Insufficient Credits!**")
            st.error("💳 You need more OpenRouter credits to use this model.")
            st.error("🔗 Visit: https://openrouter.ai/settings/credits")
            st.warning("💡 Try using a smaller model like 'mistralai/mistral-7b-instruct'")
        else:
            st.error(f"❌ Failed to connect to OpenRouter: {error_msg}")
        return
        
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    store = {}

    # Enhanced prompt with structured response format
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Use the following pieces of context to answer the question at the end.
                  If you don't know the answer, just say that you don't know, don't try to make up an answer.

                  Context: {context}

                  Please provide a structured response in the following JSON format:
                  {{
                    "answer": "Your main answer here",
                    "confidence": 0.8,
                    "reasoning": "Brief explanation of how you arrived at the answer",
                    "data_used": ["key data points from context"]
                  }}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # Use Pydantic parser for structured responses
    parser = PydanticOutputParser(pydantic_object=CSVResponse)
    runnable = prompt | llm | parser
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle chat input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        try:
            # Get context from vectorstore
            context_results = vectorstore.similarity_search(prompt, k=6)
            context = "\n\n".join(doc.page_content for doc in context_results)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Use streaming response
                try:
                    # Get structured response from model
                    response = with_message_history.invoke(
                        {"context": context, "input": prompt},
                        config={"configurable": {"session_id": "abc123"}}
                    )

                    if isinstance(response, CSVResponse):
                        # Format structured response nicely
                        full_response = f"""
**Answer:** {response.answer}

**Confidence:** {response.confidence:.2f}

**Reasoning:** {response.reasoning}
"""
                        if response.data_used:
                            full_response += f"\n**Data Used:**\n" + "\n".join(f"• {item}" for item in response.data_used)
                    elif hasattr(response, 'content'):
                        full_response = response.content
                    else:
                        full_response = str(response)

                    # Display the response
                    message_placeholder.markdown(full_response)

                    # Show raw JSON in expander for debugging
                    with st.expander("🔍 Raw Response (Debug)"):
                        if isinstance(response, CSVResponse):
                            st.json({
                                "answer": response.answer,
                                "confidence": response.confidence,
                                "reasoning": response.reasoning,
                                "data_used": response.data_used
                            })
                        else:
                            st.text(str(response))

                except Exception as stream_error:
                    st.error(f"❌ Response parsing error: {stream_error}")
                    # Fallback to basic response
                    try:
                        response = with_message_history.invoke(
                            {"context": context, "input": prompt},
                            config={"configurable": {"session_id": "abc123"}}
                        )
                        full_response = response.content if hasattr(response, 'content') else str(response)
                        message_placeholder.markdown(f"**Basic Response:**\n\n{full_response}")
                    except Exception as fallback_error:
                        st.error(f"❌ Complete failure: {fallback_error}")
                        full_response = "Sorry, I encountered an error processing your request. Please try rephrasing your question."

                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ Error processing your question: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error processing your request."})

    # Handle reset button
    if reset:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        st.rerun()

def summary(model_name, temperature, top_p, user_api_key):
    st.write("# Summary of CSV")
    st.write("Upload your document here:")
    uploaded_file = st.file_uploader("Upload source document", type="csv", label_visibility="collapsed")

    if not user_api_key or user_api_key == "":
        st.error("❌ Please enter your OpenRouter API key in the sidebar to use this functionality.")
        return

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap=100)
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()
            texts = text_splitter.split_documents(data)
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
            texts = text_splitter.split_documents(data)

        os.remove(tmp_file_path)
        
        st.info(f"📄 Loaded {len(texts)} text chunks from your CSV file")
        
        gen_sum = st.button("Generate Summary")
        if gen_sum:
            with st.spinner("Generating summary... This may take a moment."):
                try:
                    # Reduce max_tokens to fit within credit limit
                    max_tokens = 1000

                    llm = ChatOpenAI(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        base_url="https://openrouter.ai/api/v1",
                        api_key=user_api_key,
                        default_headers={
                            "HTTP-Referer": "https://localhost:8501",
                            "X-Title": "CSV AI App"
                        }
                    )
                    st.success(f"🤖 Connected to {model_name} for summarization (max_tokens: {max_tokens})")
                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type="map_reduce",
                        return_intermediate_steps=True,
                        input_key="input_documents",
                        output_key="output_text",
                    )
                    result = chain({"input_documents": texts}, return_only_outputs=True)
                    st.success("✅ Summary generated successfully!")
                    st.markdown("### 📋 Summary:")
                    st.write(result["output_text"])
                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
                    st.info("💡 Try using a smaller CSV file or check your API key.")

def analyze(temperature, model_name, user_api_key):
    st.write("# Analyze CSV")
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")

    if not user_api_key or user_api_key == "":
        st.error("❌ Please enter your OpenRouter API key in the sidebar to use this functionality.")
        return

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        df = pd.read_csv(tmp_file_path)
        os.remove(tmp_file_path)
        
        st.info(f"📊 CSV loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        with st.expander("📋 Column Information"):
            st.write("**Columns:**", list(df.columns))
            st.write("**Data Types:**")
            st.write(df.dtypes)
        
        try:
            # Reduce max_tokens to fit within credit limit
            max_tokens = 1000

            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url="https://openrouter.ai/api/v1",
                api_key=user_api_key,
                default_headers={
                    "HTTP-Referer": "https://localhost:8501",
                    "X-Title": "CSV AI App"
                }
            )
            st.success(f"🤖 Connected to {model_name} for analysis (max_tokens: {max_tokens})")
        except Exception as e:
            st.error(f"❌ Failed to connect to OpenRouter: {str(e)}")
            return
        
        try:
            agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                agent_type="openai-tools", 
                verbose=True,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                prefix="""
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
When creating visualizations:
1. Always use matplotlib.pyplot as plt
2. Always call plt.show() after creating plots
3. Use st.pyplot() to display plots in Streamlit
4. For better display, set figure size with plt.figure(figsize=(10, 6))

You should use the tools below to answer the question posed of you:
"""
            )
        except Exception as e:
            st.error(f"❌ Error creating agent: {str(e)}")
            st.stop()

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you analyze your CSV data? I can create visualizations, calculate statistics, and answer questions about your data."}]
            
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(placeholder="Try: 'Create a histogram of column X' or 'Show correlation between columns'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            try:
                with st.spinner("Analyzing..."):
                    import matplotlib.pyplot as plt
                    
                    enhanced_prompt = f"""
{prompt}

Important instructions for visualizations:
- Import matplotlib.pyplot as plt if creating plots
- Set figure size: plt.figure(figsize=(10, 6))
- After creating plots, use st.pyplot(plt.gcf()) to display in Streamlit
- Then call plt.close() to clear the figure
- Always show the plot using these Streamlit commands
"""
                    
                    msg = agent.invoke({"input": enhanced_prompt, "chat_history": st.session_state.messages})
                    
                    if plt.get_fignums():
                        st.pyplot(plt.gcf())
                        plt.close()
                    
                st.session_state.messages.append({"role": "assistant", "content": msg["output"]})
                st.chat_message("assistant").write(msg["output"])
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.chat_message("assistant").write(error_msg)
                
        if reset:
            st.session_state["messages"] = []
            
        st.sidebar.markdown("---")
        st.sidebar.warning(
            "⚠️ **Security Notice**: This feature executes Python code to analyze your data. "
            "Only use with trusted CSV files."
        )
        
        st.sidebar.markdown("---")
        st.sidebar.success(
            "💡 **Try these commands**:\n"
            "- 'Show the first 5 rows'\n"
            "- 'Create a histogram of [column]'\n"
            "- 'Calculate correlation matrix'\n"
            "- 'Plot [column1] vs [column2]'\n"
            "- 'Show summary statistics'"
        )

# Main App
def main():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>🧠 CSV AI</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>⚡️ Interacting, Analyzing and Summarizing CSV Files!</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Check for OpenRouter API key
    if os.path.exists(".env") and os.environ.get("OPENROUTER_API_KEY"):
        user_api_key = os.environ["OPENROUTER_API_KEY"]
        st.success("OpenRouter API key loaded from .env", icon="🚀")
    else:
        user_api_key = st.sidebar.text_input(
            label="#### Enter OpenRouter API key 👇", 
            placeholder="Paste your OpenRouter API key, sk-or-v1-...", 
            type="password", 
            key="openrouter_api_key"
        )
        if user_api_key:
            st.sidebar.success("OpenRouter API key loaded", icon="🚀")

    # OpenRouter model options
    MODEL_OPTIONS = [
        "x-ai/grok-4-fast:free",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-haiku-20240307",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "google/gemini-pro-1.5-latest",
        "mistralai/mistral-7b-instruct"
    ]
    
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.9
    TEMPERATURE_STEP = 0.01
    
    model_name = st.sidebar.selectbox(
        label="Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("x-ai/grok-4-fast:free")  # Set Grok as default
    )
    top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 1.0, 0.1)
    temperature = st.sidebar.slider(
                label="Temperature",
                min_value=TEMPERATURE_MIN_VALUE,
                max_value=TEMPERATURE_MAX_VALUE,
                value=TEMPERATURE_DEFAULT_VALUE,
                step=TEMPERATURE_STEP,)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Setup**:\n"
        "- **OpenRouter API**: For chat models ([openrouter.ai](https://openrouter.ai))\n"
        "- **Embeddings**: Free HuggingFace model (no API key needed!)\n\n"
        "🚀 **Using**: `sentence-transformers/all-MiniLM-L6-v2`\n"
        "🎯 **Free Model**: `x-ai/grok-4-fast:free` (default)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.warning(
        "⚠️ **Dependencies**: Make sure you have:\n"
        "```\n"
        "pip install langchain-openai langchain-community\n"
        "pip install sentence-transformers\n"
        "pip install --upgrade pydantic\n"
        "```"
    )

    functions = [
        "home",
        "Chat with CSV",
        "Analyze CSV",
    ]
    
    selected_function = st.selectbox("Select a functionality", functions)
    
    if selected_function != "home" and not user_api_key:
        st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to use this functionality.")
        return
    
    if selected_function == "home":
        home_page()
    elif selected_function == "Chat with CSV":
        chat(temperature=temperature, model_name=model_name, user_api_key=user_api_key)
    elif selected_function == "Analyze CSV":
        analyze(temperature=temperature, model_name=model_name, user_api_key=user_api_key)
    else:
        st.warning("You haven't selected any AI Functionality!!")

if __name__ == "__main__":
    main()