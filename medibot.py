import os
import streamlit as st
import time
from utils.vectorstore_utils import VectorStoreManager
from create_memory_for_llm import ensure_pdf_directory, process_and_store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

try:
    import numpy as np
except ImportError:
    st.error("NumPy is not installed. Please install NumPy with 'pip install numpy<2'.")

DB_FAISS_PATH="vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def set_page_config():
    st.set_page_config(
        page_title="MediAI Assistant",
        page_icon="👨‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""<style>
        [data-testid="stSidebarNav"] { background-image: url("https://img.icons8.com/material/96/000000/caduceus.png"); background-repeat: no-repeat; padding-top: 100px; background-position: 20px 20px; }
        .main { padding: 2rem; max-width: 1200px; margin: 0 auto; background-color: #1e1e1e; color: #ffffff; }
        h1, h2, h3 { color: #1E88E5; font-family: 'Segoe UI', sans-serif; }
        .stCard { border-radius: 15px; padding: 1.5rem; margin: 1rem 0; background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .file-list { background: linear-gradient(145deg, #2e2e2e, #3e3e3e); padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 1rem 0; }
        .file-item { display: flex; align-items: center; padding: 0.5rem; border-bottom: 1px solid #444; transition: all 0.3s ease; }
        .file-item:hover { background: rgba(30, 136, 229, 0.1); }
        .status-badge { padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.8rem; font-weight: 500; }
        .status-success { background: #4CAF50; color: white; }
        .stProgress > div > div { background-color: #1E88E5; }
        .stButton>button { width: 100%; border-radius: 25px; transition: all 0.3s ease; border: none; padding: 0.5rem 1rem; background: linear-gradient(45deg, #1E88E5, #1976D2); color: white; }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); }
        </style>""", unsafe_allow_html=True)

def sidebar_navigation():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hospital-2.png")
        st.title("🏥 MediAI")
        st.markdown("---")
        selected = st.radio("Go to", ["👨‍⚕️ Document Management", "🤖 Chat Assistant", "🔬 Knowledge Base", "ℹ️ About"])
        st.markdown("---")
        with st.expander("⚙️ Settings", expanded=False):
            st.checkbox("🌙 Dark Mode (Beta)")
            st.slider("📝 Chunk Size", 100, 1000, 500)
        st.markdown("---")
        st.markdown("### 📊 System Status")
    return selected

def document_management():
    st.header("📚 Document Management")
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.expander("ℹ️ Quick Guide", expanded=False):
            st.markdown("""### Getting Started
            1. 📄 Upload your medical documents
            2. 🔄 Watch as they're automatically processed
            3. 📊 View statistics and analysis
            4. 🤖 Use the chatbot to query the knowledge base""")
        st.markdown("""<div class='stCard'><h3>📄 Upload Medical Documents</h3><p>Supported formats: PDF</p></div>""", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload your files", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            pdf_folder = ensure_pdf_directory()
            for uploaded_file in uploaded_files:
                save_path = os.path.join(pdf_folder, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.success(f"✅ Saved: {uploaded_file.name}")
            process_and_store()
    with col2:
        st.subheader("📊 Analytics")
        pdf_folder = ensure_pdf_directory()
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        excluded_files = [
            "Gale Encyclopedia of Medicine Vol. 2 (C-F).pdf",
            "Gale Encyclopedia of Medicine Vol. 1.pdf",
            "Gale Encyclopedia of Medicine Vol. 5 (T-Z).pdf"
        ]
        pdf_files = [f for f in pdf_files if f not in excluded_files]
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Documents", len(pdf_files), "📚")
        with col_b:
            st.metric("Pages", len(pdf_files) * 5, "📄")
        if pdf_files:
            st.markdown("<div class='file-list'>", unsafe_allow_html=True)
            for pdf_file in pdf_files:
                st.markdown(f"""<div class='file-item'><span>📄 {pdf_file}</span><span class='status-badge status-success'>Processed</span></div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

def test_vectorstore():
    st.header("🔍 Vector Store Analysis")
    manager = VectorStoreManager()
    stats = manager.get_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistics")
        if stats['status'] == 'loaded':
            st.metric("Total Chunks", stats.get('total_documents', 0))
            st.metric("Unique Sources", stats.get('unique_sources', 0))
        else:
            st.error("Vector store not loaded")
    with col2:
        st.subheader("Sample Content")
        if stats['status'] == 'loaded':
            with st.expander("View document samples", expanded=False):
                manager.inspect_documents(limit=2)

def about_page():
    st.header("ℹ️ About Medical Knowledge Base")
    st.markdown("""<div class='stCard'><h3>🏥 Medical Knowledge Management System</h3><p>Version 1.0.0</p><p>A sophisticated tool for managing and querying medical documentation.</p></div>""", unsafe_allow_html=True)

def chatbot_interface():
    st.header("🤖 Medical Chatbot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    prompt = st.chat_input("Ask me anything about the medical documents...")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return
            CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question. If you dont know the answer, just say that you dont know, dont try to make up an answer. Dont provide anything out of the given context
            Context: {context}
            Question: {question}
            Start the answer directly. No small talk please."""
            HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
            HF_TOKEN=os.environ.get("HF_TOKEN")
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response = qa_chain.invoke({'query':prompt})
            result = response["result"]
            result_markdown = f"""{result}"""
            st.chat_message('assistant').markdown(result_markdown)
            st.session_state.messages.append({'role':'assistant', 'content': result_markdown})
        except Exception as e:
            st.error(f"Error: {str(e)}")

def main():
    load_dotenv(find_dotenv())
    set_page_config()
    st.markdown("""<div style='text-align: center; padding: 1rem;'><h1>👨‍⚕️ MediAI Assistant</h1><p style='color: #666;'>Your Intelligent Medical Knowledge Assistant</p></div>""", unsafe_allow_html=True)
    selected = sidebar_navigation()
    if selected == "👨‍⚕️ Document Management":
        document_management()
    elif selected == "🤖 Chat Assistant":
        chatbot_interface()
    elif selected == "🔬 Knowledge Base":
        test_vectorstore()
    else:
        about_page()

if __name__ == "__main__":
    main()
    # Ensure Streamlit server runs
    st._is_running_with_streamlit = True