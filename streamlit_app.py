"""
Streamlit application for Advanced RAG Chatbot
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper function to run async code in Streamlit
def run_async(coro):
    """Run async coroutine in Streamlit context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


# Local imports - with error handling
try:
    from config import PERFORMANCE_METRICS, config
    from document_processor import DocumentProcessor
    from ml_utils import PerformanceMonitor
    from rag_engine import RAGConfig, RAGEngine, RAGResponse
    from vector_store import Document

    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)
    logger.error(f"Import error: {e}")

# Streamlit page configuration
st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        flex-direction: column;
    }

    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }

    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }

    .source-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

    if "performance_data" not in st.session_state:
        st.session_state.performance_data = {
            "response_times": [],
            "confidence_scores": [],
            "query_count": 0,
            "timestamps": [],
        }

    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def initialize_rag_engine():
    """Initialize RAG engine with API key"""
    if not IMPORTS_SUCCESSFUL:
        st.error(f"Import error: {IMPORT_ERROR}")
        return False

    api_key = st.session_state.get("gemini_api_key", "")

    if not api_key:
        st.error("Please enter your Gemini API key to continue.")
        return False

    try:
        # Create RAG configuration
        rag_config = RAGConfig(
            model_name=st.session_state.get("model_name", "gemini-1.5-flash"),
            temperature=st.session_state.get("temperature", 0.7),
            max_tokens=st.session_state.get("max_tokens", 2048),
            top_k=st.session_state.get("top_k", 5),
            similarity_threshold=st.session_state.get("similarity_threshold", 0.3),
            enable_reranking=st.session_state.get("enable_reranking", True),
            enable_query_expansion=st.session_state.get("enable_query_expansion", True),
            enable_caching=st.session_state.get("enable_caching", True),
        )

        # Initialize RAG engine
        with st.spinner("Initializing RAG engine..."):
            st.session_state.rag_engine = RAGEngine(
                api_key=api_key, rag_config=rag_config
            )
            st.session_state.initialized = True

        st.success("✅ RAG engine initialized successfully!")
        return True

    except Exception as e:
        st.error(f"❌ Error initializing RAG engine: {str(e)}")
        logger.error(f"RAG engine initialization error: {e}", exc_info=True)
        return False


def sidebar_configuration():
    """Sidebar for configuration and settings"""
    st.sidebar.title("🔧 Configuration")

    # Check imports
    if not IMPORTS_SUCCESSFUL:
        st.sidebar.error(f"Import Error: {IMPORT_ERROR}")
        st.sidebar.info("Please check your dependencies and configuration files.")
        return

    # API Key input
    st.sidebar.subheader("API Configuration")
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get("gemini_api_key", ""),
        help="Enter your Google Gemini API key",
    )
    st.session_state.gemini_api_key = api_key

    # Model Configuration
    st.sidebar.subheader("Model Settings")

    model_name = st.sidebar.selectbox(
        "Model", options=["gemini-2.5-flash", "gemini-1.5-pro"], index=0
    )
    st.session_state.model_name = model_name

    temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1
    )
    st.session_state.temperature = temperature

    # Initialize button
    if st.sidebar.button("Initialize RAG Engine", type="primary"):
        if initialize_rag_engine():
            st.rerun()

    # Status indicator
    if st.session_state.initialized:
        st.sidebar.success("✅ RAG Engine Active")
    else:
        st.sidebar.warning("⚠️ RAG Engine Not Initialized")

    # System actions
    st.sidebar.subheader("Actions")

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Add this test button in sidebar
    if st.sidebar.button("Test API Key"):
        try:
            import google.generativeai as genai

            genai.configure(api_key=st.session_state.gemini_api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content("Say 'API key works!'")
            st.sidebar.success(f"✅ API Response: {response.text}")
        except Exception as e:
            st.sidebar.error(f"❌ API Error: {e}")


def chat_interface():
    """Main chat interface"""
    st.header("💬 Chat with Your Documents")

    if not IMPORTS_SUCCESSFUL:
        st.error(f"Cannot load chat interface due to import error: {IMPORT_ERROR}")
        return

    if not st.session_state.rag_engine:
        st.warning(
            "⚠️ Please initialize the RAG engine first using the sidebar configuration."
        )
        st.info(
            "👈 Enter your Gemini API key in the sidebar and click 'Initialize RAG Engine'"
        )
        return

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

                # Show metadata in expander
                if "metadata" in message:
                    with st.expander("Details"):
                        metadata = message["metadata"]
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "Confidence", f"{metadata.get('confidence', 0):.2%}"
                            )
                            st.metric(
                                "Processing Time",
                                f"{metadata.get('processing_time', 0):.2f}s",
                            )

                        with col2:
                            st.metric("Tokens", metadata.get("tokens_used", 0))
                            st.metric(
                                "Cache Hit",
                                "Yes" if metadata.get("cache_hit") else "No",
                            )

    # Chat input
    user_input = st.chat_input("Ask a question...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "timestamp": datetime.now()}
        )

        # Generate response with detailed debugging
        with st.spinner("🤔 Thinking..."):
            try:
                # DEBUG: Step 1 - Check vector store search
                import traceback

                st.write("🔍 Searching documents...")
                test_search = (
                    st.session_state.rag_engine.vector_store.similarity_search(
                        user_input, k=5
                    )
                )
                st.write(f"✅ Found {len(test_search)} documents")

                if test_search:
                    with st.expander("📄 Retrieved Documents (Debug)"):
                        for i, doc in enumerate(test_search[:3]):
                            st.write(f"**Document {i+1}** - Score: {doc.score:.3f}")
                            st.write(doc.content[:150] + "...")
                            st.divider()

                # DEBUG: Step 2 - Query RAG engine
                st.write("🤖 Generating response...")
                response = run_async(st.session_state.rag_engine.query(user_input))
                st.write(f"✅ Done! Confidence: {response.confidence:.2%}")

                # Add assistant message
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "timestamp": datetime.now(),
                        "metadata": {
                            "confidence": response.confidence,
                            "processing_time": response.processing_time,
                            "tokens_used": response.tokens_used,
                            "cache_hit": response.cache_hit,
                        },
                    }
                )

                # Update performance data
                st.session_state.performance_data["response_times"].append(
                    response.processing_time
                )
                st.session_state.performance_data["confidence_scores"].append(
                    response.confidence
                )
                st.session_state.performance_data["query_count"] += 1

            except Exception as e:
                st.error(f"❌ ERROR: {type(e).__name__}: {str(e)}")

                # Show full traceback
                import traceback

                with st.expander("🐛 Full Error Details"):
                    st.code(traceback.format_exc())

                logger.error(f"Chat error: {e}", exc_info=True)

        st.rerun()


def document_management_page():
    """Document management interface"""
    st.header("📚 Document Management")

    if not IMPORTS_SUCCESSFUL:
        st.error(f"Cannot load document management: {IMPORT_ERROR}")
        return

    if not st.session_state.rag_engine:
        st.warning("⚠️ Please initialize the RAG engine first.")
        return

    # File upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "md"],
        help="Upload documents to add to the knowledge base",
    )

    if uploaded_files:
        st.write(f"📄 Selected {len(uploaded_files)} files")

        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing..."):
                temp_dir = "./temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)

                file_paths = []
                try:
                    # Save files
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(file_path)

                    # Process
                    result = run_async(
                        st.session_state.rag_engine.add_documents(file_paths)
                    )

                    if result.get("success"):
                        st.success(
                            f"✅ Added {result.get('documents_added', 0)} documents!"
                        )
                        st.session_state.documents_loaded = True
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Document processing error: {e}", exc_info=True)

                finally:
                    # Cleanup
                    for fp in file_paths:
                        try:
                            os.remove(fp)
                        except:
                            pass

    # Status
    if st.session_state.documents_loaded:
        st.success("✅ Documents loaded")
    else:
        st.info("ℹ️ No documents loaded yet")


def analytics_page():
    """Analytics and performance"""
    st.header("📊 Analytics")

    if not st.session_state.performance_data["query_count"]:
        st.info("No data yet. Start chatting to see analytics!")
        return

    data = st.session_state.performance_data

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_time = sum(data["response_times"]) / len(data["response_times"])
        st.metric("Avg Response Time", f"{avg_time:.2f}s")

    with col2:
        avg_conf = sum(data["confidence_scores"]) / len(data["confidence_scores"])
        st.metric("Avg Confidence", f"{avg_conf:.2%}")

    with col3:
        st.metric("Total Queries", data["query_count"])

    # Charts
    if len(data["response_times"]) > 1:
        df = pd.DataFrame(
            {
                "Query": range(1, len(data["response_times"]) + 1),
                "Response Time": data["response_times"],
                "Confidence": data["confidence_scores"],
            }
        )

        st.subheader("Response Time Trend")
        fig = px.line(df, x="Query", y="Response Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confidence Distribution")
        fig2 = px.histogram(df, x="Confidence", nbins=20)
        st.plotly_chart(fig2, use_container_width=True)


def main():
    """Main application"""
    # Initialize
    init_session_state()

    # Sidebar
    sidebar_configuration()

    # Header
    st.markdown(
        '<h1 class="main-header">🤖 Advanced RAG Chatbot</h1>', unsafe_allow_html=True
    )

    # Show import error prominently if exists
    if not IMPORTS_SUCCESSFUL:
        st.error(f"⚠️ Import Error: {IMPORT_ERROR}")
        st.info(
            "Please check that all dependencies are installed and configuration files exist."
        )
        st.code("pip install -r requirements.txt", language="bash")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Documents", "📊 Analytics"])

    with tab1:
        chat_interface()

    with tab2:
        document_management_page()

    with tab3:
        analytics_page()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Advanced RAG Chatbot v2.0</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)
        st.code(str(e))
