import os
import chainlit as cl
import numpy as np

from utils.vectorstore_utils import VectorStoreManager
from create_memory_for_llm import ensure_pdf_directory, process_and_store
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from dotenv import load_dotenv, find_dotenv

DB_FAISS_PATH = "vectorstore/db_faiss"

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    client = InferenceClient(token=HF_TOKEN)
    return client

def load_all_data_files(directory):
    pdf_folder = ensure_pdf_directory()
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            save_path = os.path.join(pdf_folder, filename)
            with open(save_path, "wb") as wf:
                with open(os.path.join(directory, filename), "rb") as f:
                    wf.write(f.read())
            print(f"Loaded and saved: {filename}")
    process_and_store()

@cl.on_message
async def on_message(message):
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            await cl.Message(content="Failed to load the vector store").send()
            return
        CUSTOM_PROMPT_TEMPLATE = "Use the pieces of information provided in the context to answer user's question. If you dont know the answer, just say that you dont know, dont try to make up an answer. Dont provide anything out of the given context\nContext: {context}\nQuestion: {question}\nStart the answer directly. No small talk please."
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        client = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)
        response = client.chat_completion(
            model=HUGGINGFACE_REPO_ID,  
            messages=[
                {"role": "system", "content": CUSTOM_PROMPT_TEMPLATE},
                {"role": "user", "content": str(message)}
            ]
        )
        result = response["choices"][0]["message"]["content"]
        await cl.Message(content=result).send()
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

def main():
    load_dotenv(find_dotenv())
    load_all_data_files("path/to/your/data/files")  # Add this line to load all data files
    cl.run()

if __name__ == "__main__":
    main()
    cl._is_running_with_chainlit = True