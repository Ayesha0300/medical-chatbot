import os
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

class VectorStoreManager:
    def __init__(self, path: str = "vectorstore/db_faiss"):
        self.path = path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None
        self._load_vectorstore()

    def _load_vectorstore(self) -> None:
        """Load the FAISS vectorstore from disk"""
        try:
            if os.path.exists(self.path):
                self.db = FAISS.load_local(self.path, self.embedding_model)
                print(f"Successfully loaded vectorstore from {self.path}")
            else:
                print(f"Warning: No vectorstore found at {self.path}")
        except Exception as e:
            print(f"Error loading vectorstore: {str(e)}")

    def search_similar(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents"""
        if not self.db:
            print("No vectorstore loaded")
            return []
        return self.db.similarity_search(query, k=k)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vectorstore"""
        if not self.db:
            return {"status": "No vectorstore loaded"}
        
        try:
            docs = self.db.similarity_search("", k=1000)
            sources = set(doc.metadata.get('source', 'unknown') for doc in docs)
            
            return {
                "total_documents": len(docs),
                "unique_sources": len(sources),
                "sources": list(sources),
                "status": "loaded"
            }
        except Exception as e:
            return {"status": f"Error: {str(e)}"}

    def inspect_documents(self, start: int = 0, limit: int = 5) -> None:
        """Print detailed information about documents"""
        if not self.db:
            print("No vectorstore loaded")
            return

        docs = self.db.similarity_search("", k=start + limit)
        
        for i, doc in enumerate(docs[start:start+limit], start + 1):
            print(f"\n{'='*50}")
            print(f"Document {i}:")
            print(f"{'='*50}")
            print(f"Content Preview: {doc.page_content[:300]}...")
            print(f"\nMetadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")

def main():
    manager = VectorStoreManager()
    
    # Print statistics
    print("\nVectorstore Statistics:")
    print("="*50)
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Inspect some documents
    print("\nDocument Inspection:")
    print("="*50)
    manager.inspect_documents(limit=3)
    
    # Example search
    print("\nExample Search:")
    print("="*50)
    query = "What is diabetes?"
    results = manager.search_similar(query, k=2)
    print(f"Results for query: '{query}'")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()
