# test_loader.py
from rag.document_loader import load_documents

chunks = load_documents("docs")

print(f"\nTotal chunks: {len(chunks)}")
print(f"\nFirst chunk:")
print(f"  Source : {chunks[0].source}")
print(f"  ID     : {chunks[0].chunk_id}")
print(f"  Text   : {chunks[0].text[:100]}...")
