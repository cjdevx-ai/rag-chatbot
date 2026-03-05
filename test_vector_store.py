# test_vector_store.py
from rag.document_loader import load_documents
from rag.embedder import embed_chunks, embed_query
from rag.vector_store import VectorStore

# Load and embed chunks
print("Loading documents...")
chunks = load_documents("docs")

print("Embedding chunks...")
vectors = embed_chunks([c.text for c in chunks])

# Build the store
store = VectorStore(dim=vectors.shape[1])
store.add(chunks, vectors)

# Search
print("\nSearching for: 'What is quantum superposition?'")
query_vec = embed_query("What is quantum superposition?")
results = store.search(query_vec, k=3)

for chunk, score in results:
    print(f"\n  Score  : {score:.3f}")
    print(f"  Source : {chunk.source}")
    print(f"  Text   : {chunk.text[:120]}...")