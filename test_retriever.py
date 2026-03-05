# test_retriever.py
from rag.document_loader import load_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever

# Build the pipeline up to retrieval
print("Loading and embedding documents...")
chunks = load_documents("docs")

# Initialize shared embedder
embedder = Embedder()
vectors = embedder.embed([c.text for c in chunks])

store = VectorStore(dim=vectors.shape[1])
store.add(chunks, vectors)

retriever = Retriever(embedder, store, top_k=3, min_score=0.1)

# Try a few queries
queries = [
    "How does the NOVA system detect vehicle speed?",
    "What is NailSpect and how does it estimate hemoglobin?",
    "Tell me about the campus environmental monitoring system using LoRa.",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = retriever.retrieve(query)

    if not results:
        print("  No relevant chunks found above min_score threshold.")
    else:
        for chunk, score in results:
            print(f"  [{score:.3f}] {chunk.source} → {chunk.text[:100]}...")