import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

def ensure_pdf_directory():
    pdf_folder = "data"
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"Created new PDF directory at: {pdf_folder}")
    return pdf_folder

def load_pdf_files(data_path="data/"):
    ensure_pdf_directory()
    loader = DirectoryLoader(data_path,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF files")
    return documents

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                  chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Created {len(text_chunks)} text chunks")
    return text_chunks

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

def process_and_store():
    # Load and process PDFs
    documents = load_pdf_files()
    if not documents:
        print("No documents found to process")
        return
    
    # Create chunks and embeddings
    text_chunks = create_chunks(documents)
    embedding_model = get_embedding_model()
    
    # Store in FAISS
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    process_and_store()