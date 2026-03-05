import os
from dotenv import load_dotenv
from rag.document_loader import load_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator

load_dotenv()

# Build the full pipeline
print("Setting up pipeline...")
chunks = load_documents("docs")
embedder = Embedder()
vectors = embedder.embed([c.text for c in chunks])

store = VectorStore(dim=vectors.shape[1])
store.add(chunks, vectors)

api_key = os.getenv("GEMINI_API_KEY")
retriever = Retriever(embedder, store, top_k=3, min_score=0.1)
generator = Generator(api_key=api_key, model="gemini-flash-latest")

# Test a real question against your docs
query = "How does the NOVA system detect vehicle speed?"
print(f"\nQuestion: {query}\n")

retrieved = retriever.retrieve(query)
# Pass empty history for testing
answer = generator.generate(query, retrieved, history=[])

print("Answer:")
print(answer)
print("\nSources used:")
for chunk, score in retrieved:
    print(f"  [{score:.3f}] {chunk.source}")