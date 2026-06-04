# """
# RAG Pipeline: Load PDFs, chunk, embed, and store in vector database
# """
# from pathlib import Path
# import ollama
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_community.document_loaders import OnlinePDFLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain_classic.retrievers.multi_query import MultiQueryRetriever 

# class RAGPipeline:
#     def __init__(self, data_folder: str):
#         """
#         Initialize RAG pipeline.
        
#         Args:
#             data_folder: Path to folder containing PDFs
#             embedding_model: HuggingFace model for embeddings
#         """
#         self.data_folder = data_folder
#         self.documents = []
#         self.vector_store = None

#     def load_pdfs(self):
#         """
#         Load all PDFs from data folder and stores them in the class's document list.
#         """

#         print(f"Loading PDFs from {self.data_folder}...")
#         pdf_files = list(Path(self.data_folder).glob("*.pdf"))
#         print(f"Found {len(pdf_files)} PDF files")

#         docs = []
#         for file in pdf_files:
#             loader = UnstructuredPDFLoader(file_path=str(file))
#             data = loader.load()
#             print(f"Done loading {file.name}...")

#             for doc in data:
#                 doc.metadata["source"] = file.name
#             docs.extend(data)

#         self.documents = docs
#         print(f"Total documents loaded: {len(self.documents)}")
#         return docs

#     def chunk_documents(self):
#         """
#         Split documents into chunks and stores them in the class's document list.
#         """

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300) # each chunk is 1200 chars with 300 char overlap (greater overlap the better because it includes context)
#         chunks = text_splitter.split_documents(self.documents)
#         print("Done splitting...")
#         self.documents = chunks
#         return chunks
    
#     def create_embeddings_and_store(self, vector_db_path: str = "vector_db") -> Chroma:
#         """
#         Create embeddings and store in Chroma vector database.
#         """

#         ollama.pull("nomic-embed-text")

#         self.vector_store = Chroma.from_documents(
#             documents=self.documents,
#             embedding=OllamaEmbeddings(model="nomic-embed-text"),
#             collection_name="rag_collection")
        
#         self.vector_store.save_local(vector_db_path)
#         print(f"Vector database saved to {vector_db_path}")
        
#         print("Done adding to vector database...")
#         return self.vector_store
    
#     def query_and_retrieve(self, query_text: str, top_k: int = 3, vector_db: Chroma = None):
#         """
#         Search vector database for similar documents and answer query using RAG approach.

#         Args:
#             query_text: Search query
#             top_k: Number of results to return
#             vector_db: Chroma vector database instance
        
#         """
#         llm = ChatOllama(model="llama3.2")
#         QueryPrompt = PromptTemplate(
#             input_variables=["question"],
#             template="Given the following question, retrieve the most relevant documents from the vector database: {question}"
#         )

#         retriever = MultiQueryRetriever.from_llm(
#             vector_db.as_retriever(), llm, prompt=QueryPrompt
#         )

#         # RAG prompt
#         template = """Answer the question based on the following retrieved documents. If you don't know the answer, say you don't know.
#         {context}
#         Question: {question}
#         """

#         prompt = ChatPromptTemplate.from_template(template)

#         chain = (
#             {"context": retriever, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )

#         res = chain.invoke(query_text)
#         return res

    

# def main():
#     """Example usage of RAG pipeline."""
    
#     # Initialize pipeline
#     data_folder = r"C:\Users\ik_ad\DSA-Tutor\DSA-Tutor\Data"
#     pipeline = RAGPipeline(data_folder)
    
#     # Step 1: Load PDFs
#     pipeline.load_pdfs()

#     # Step 2: Chunk documents
#     pipeline.chunk_documents()

#     # Step 3: Create embeddings and store in vector database
#     vector_db = pipeline.create_embeddings_and_store()

#     # Step 4: Send query and retrieve relevant information
#     res = pipeline.query_and_retrieve("What is dynamic programming?", top_k=2, vector_db=vector_db)
#     print("\n" + "="*60)
#     print("Query Result:\n" + res)



# if __name__ == "__main__":
#     main()




from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Document loading and processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class RAGPipeline:
    def __init__(
        self,
        data_folder: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generation_model: str = "microsoft/Phi-4-mini-instruct",
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            data_folder: Path to folder containing PDFs
            embedding_model: HuggingFace model for embeddings
            generation_model: HuggingFace instruction model used to answer questions
            vector_store: holds the FAISS vector database instance after creation or loading
            self.documents: holds the list of Document objects after loading and chunking
        """
        self.data_folder = Path(data_folder)
        self.embedding_model = embedding_model
        self.generation_model = generation_model
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
            torch_dtype="auto",
            attn_implementation="eager",
        )
        self.generator.eval()

    def generate_answer(
        self,
        query_text: str,
        top_k: int = 3,
        max_new_tokens: int = 300,
    ) -> str:
        """Retrieve relevant chunks and use Phi-4 Mini to answer the question."""
        results = self.query(query_text, top_k=top_k)
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc, _ in results
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a DSA tutor. Answer the user's question using only the "
                    "provided context. If the context does not contain the answer, "
                    "say that you do not know. Cite source filenames when useful."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query_text}",
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
        answer = pipeline.generate_answer(query, top_k=3)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
