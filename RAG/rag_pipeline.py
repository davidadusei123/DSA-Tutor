from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Document loading and processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RAGPipeline:
    def __init__(
        self,
        data_folder: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generation_model: str = "microsoft/Phi-4-mini-instruct",
        adapter_model: str = "models/phi-4-mini-dsa-adapter",
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            data_folder: Path to folder containing PDFs
            embedding_model: HuggingFace model for embeddings
            generation_model: HuggingFace instruction model used to answer questions
            adapter_model: Path to the fine-tuned LoRA adapter
            vector_store: holds the FAISS vector database instance after creation or loading
            self.documents: holds the list of Document objects after loading and chunking
        """
        self.data_folder = Path(data_folder)
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        adapter_path = Path(adapter_model)
        self.adapter_model = (
            adapter_path
            if adapter_path.is_absolute()
            else PROJECT_ROOT / adapter_path
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.tokenizer = None
        self.generator = None
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

    def _load_generator(self) -> None:
        """Load the generation model only when an answer is requested."""
        if self.generator is not None:
            return

        print(f"Loading generation model {self.generation_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.generation_model)
        self.generator = AutoModelForCausalLM.from_pretrained(
            self.generation_model,
            device_map="auto",
            dtype="auto",
            attn_implementation="eager",
        )
        if self.adapter_model.exists():
            print(f"Loading fine-tuned adapter {self.adapter_model}...")
            self.generator = PeftModel.from_pretrained(
                self.generator,
                str(self.adapter_model),
            )
        else:
            print(
                f"Fine-tuned adapter not found at {self.adapter_model}. "
                "Using the base generation model."
            )
        self.generator.eval()

    @staticmethod
    def _mode_system_prompt(mode: str) -> str:
        """Return generation instructions for the selected answer mode."""
        shared = (
            "You are a DSA tutor. Answer the student's question using only the "
            "provided retrieved context. If the context does not contain the answer, "
            "say that you do not know."
        )
        if mode == "raw_answer":
            return (
                f"{shared} Give a direct, detailed explanation. Include relevant "
                "definitions, reasoning, steps, complexity, and practical application."
            )
        if mode == "summary":
            return (
                f"{shared} Summarize the answer as clear bullet points for a student "
                "with coding experience who is still learning DSA. Avoid unnecessary "
                "jargon and explain required technical terms simply."
            )
        raise ValueError("mode must be either 'raw_answer' or 'summary'")

    def generate_answer(
        self,
        query_text: str,
        mode: str = "raw_answer",
        top_k: int = 3,
        max_new_tokens: int = 300,
    ) -> str:
        """Retrieve relevant chunks and use the fine-tuned Phi model to answer."""
        results = self.query(query_text, top_k=top_k)
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc, _ in results
        )

        messages = [
            {
                "role": "system",
                "content": self._mode_system_prompt(mode),
            },
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{context}\n\n"
                    f"Student question: {query_text}"
                ),
            },
        ]

        self._load_generator()
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.generator.device)

        with torch.inference_mode():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()
    
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
    
    # Step 4: Retrieve context and generate answers with Phi-4 Mini
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
        answer = pipeline.generate_answer(query, mode="summary", top_k=3)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
