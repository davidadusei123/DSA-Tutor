"""
RAG Pipeline: Load PDFs, chunk, embed, and store in vector database
"""

import os
from pathlib import Path
from typing import List

# Document loading and processing
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


class RAGPipeline:
    def __init__(self, data_folder: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG pipeline.
        
        Args:
            data_folder: Path to folder containing PDFs
            embedding_model: HuggingFace model for embeddings
        """
        self.data_folder = Path(data_folder)
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.documents = []
        
    def load_pdfs(self) -> List[Document]:
        """Load all PDFs from data folder."""
        print(f"Loading PDFs from {self.data_folder}...")
        
        pdf_files = list(self.data_folder.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        all_documents = []
        for pdf_file in pdf_files:
            print(f"  Loading {pdf_file.name}...")
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = pdf_file.name
            
            all_documents.extend(documents)
        
        self.documents = all_documents
        print(f"Total documents loaded: {len(self.documents)}")
        return all_documents
    
    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Overlap between chunks
        """
        print(f"\nChunking documents (size={chunk_size}, overlap={chunk_overlap})...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(self.documents)
        print(f"Total chunks created: {len(chunks)}")
        
        self.documents = chunks
        return chunks
    
    def create_embeddings_and_store(self, vector_db_path: str = "vector_db") -> FAISS:
        """
        Create embeddings and store in FAISS vector database.
        
        Args:
            vector_db_path: Path to save vector database
        """
        print(f"\nCreating embeddings and storing in vector database...")
        print(f"This may take a few minutes depending on document size...")
        
        # Create FAISS vector store from documents
        self.vector_store = FAISS.from_documents(
            self.documents,
            self.embeddings
        )
        
        # Save to disk
        self.vector_store.save_local(vector_db_path)
        print(f"Vector database saved to {vector_db_path}")
        
        return self.vector_store
    
    def query(self, query_text: str, top_k: int = 3) -> List[tuple]:
        """
        Search vector database for similar documents.
        
        Args:
            query_text: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Run create_embeddings_and_store() first.")
        
        results = self.vector_store.similarity_search_with_score(query_text, k=top_k)
        return results
    
    def load_existing_vectordb(self, vector_db_path: str = "vector_db") -> FAISS:
        """Load pre-existing vector database."""
        print(f"Loading vector database from {vector_db_path}...")
        self.vector_store = FAISS.load_local(
            vector_db_path,
            self.embeddings
        )
        print("Vector database loaded successfully")
        return self.vector_store


def main():
    """Example usage of RAG pipeline."""
    
    # Initialize pipeline
    data_folder = r"C:\Users\ik_ad\DSA-Tutor\DSA-Tutor\Data"
    pipeline = RAGPipeline(data_folder)
    
    # Step 1: Load PDFs
    pipeline.load_pdfs()
    
    # Step 2: Chunk documents
    pipeline.chunk_documents(chunk_size=500, chunk_overlap=50)
    
    # Step 3: Create embeddings and store in vector database
    pipeline.create_embeddings_and_store("vector_db")
    
    # Step 4: Query the vector database
    print("\n" + "="*60)
    print("Testing RAG Pipeline with sample queries")
    print("="*60)
    
    test_queries = [
        "What is dynamic programming?",
        "Explain binary search trees",
        "How does sorting work?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = pipeline.query(query, top_k=2)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  Result {i} (score: {score:.4f})")
            print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"    Preview: {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()
