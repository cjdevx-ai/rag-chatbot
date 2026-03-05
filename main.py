import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from rag.pipeline import RAGPipeline

# --- Page configuration ---
st.set_page_config(
    page_title="RAG Knowledge Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load env variables
load_dotenv()

# --- Custom Styling: Cool Night Blue ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
    }

    /* Chat input styling */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    
    /* Custom Chat Bubbles */
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(56, 189, 248, 0.1) !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }

    [data-testid="stChatMessageContent"] p {
        color: #e2e8f0 !important;
        line-height: 1.6;
    }

    /* User message specific tweak */
    [data-testid="stChatMessage"]:has(path[d*="M12 12c2.21"]) {
        border-left: 4px solid #38bdf8 !important;
    }

    /* Assistant message specific tweak */
    [data-testid="stChatMessage"]:has(path[d*="M20 13c0"]) {
        border-left: 4px solid #818cf8 !important;
    }

    /* Expander styling */
    .stExpander {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(56, 189, 248, 0.1) !important;
        border-radius: 10px !important;
    }

    /* Sidebar buttons */
    .stButton > button {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #7dd3fc !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4) !important;
    }

    /* Info boxes */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #e2e8f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Singleton Pipeline Manager ---
@st.cache_resource
def get_pipeline():
    # Priority: st.secrets (Cloud) -> os.getenv (Local)
    api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        # st.secrets is not available or file is missing
        pass
        
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("🔑 API Key Missing: Please set GEMINI_API_KEY in Streamlit Secrets or .env file.")
        st.stop()
    
    docs_dir = "docs"
    if not Path(docs_dir).is_dir():
        st.error(f"📁 Missing Folder: Docs directory '{docs_dir}' not found.")
        st.stop()
        
    # Return initialized pipeline
    return RAGPipeline(
        docs_dir=docs_dir,
        api_key=api_key,
        top_k=int(os.getenv("TOP_K", "3")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        embed_model=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    )

pipeline = get_pipeline()

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("Settings & Info")
    
    # Statistics
    st.subheader("Knowledge Base")
    doc_counts = {}
    for chunk in pipeline.chunks:
        doc_counts[chunk.source] = doc_counts.get(chunk.source, 0) + 1
    
    st.info(f"Total chunks indexed: **{len(pipeline.chunks)}** across **{len(doc_counts)}** files.")
    
    with st.expander("Show Indexed Files"):
        for src, count in sorted(doc_counts.items()):
            st.write(f"- {src} ({count})")
    
    st.divider()
    
    # Controls
    if st.button("Clear Chat History", use_container_width=True):
        pipeline.reset_history()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("v1.0.0 | Powered by Gemini & FAISS")

# --- Main UI ---
st.title("ECE THESIS LIBRARY ASSISTANT")
st.markdown("Ask questions about your research documents. The assistant will provide grounded answers with citations.")

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Sources"):
                for i, (chunk, score) in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}] {chunk.source}** (Similarity: {score:.2f})")
                    st.code(chunk.text[:500] + "...", language="text")

# Chat input
if prompt := st.chat_input("Enter your question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                answer, sources = pipeline.query(prompt)
                st.markdown(answer)
                
                if sources:
                    with st.expander("View Sources"):
                        for i, (chunk, score) in enumerate(sources, 1):
                            st.markdown(f"**[{i}] {chunk.source}** (Similarity: {score:.2f})")
                            st.code(chunk.text[:500] + "...", language="text")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
