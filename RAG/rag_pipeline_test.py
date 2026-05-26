from rag_pipeline import RAGPipeline

# Create pipeline
pipeline = RAGPipeline(r"C:\Users\ik_ad\DSA-Tutor\DSA-Tutor\Data")

# Option A: Build fresh
pipeline.load_pdfs()
pipeline.chunk_documents()
pipeline.create_embeddings_and_store("vector_db")

# Option B: Load existing (after first run)
pipeline.load_existing_vectordb("vector_db")

# Query
results = pipeline.query("Explain depth first search in simple terms?", top_k=3)
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}\n")