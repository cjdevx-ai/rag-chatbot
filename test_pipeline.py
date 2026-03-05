import os
from dotenv import load_dotenv
from rag.pipeline import RAGPipeline

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

pipeline = RAGPipeline(docs_dir="docs", api_key=api_key)

# Single question
answer, sources = pipeline.query("How does the NOVA system detect vehicle speed?")
print("Answer:", answer)
print("\nSources:")
for chunk, score in sources:
    print(f"  [{score:.3f}] {chunk.source}")

# Follow-up question (tests multi-turn)
answer2, _ = pipeline.query("Can you elaborate on the sensor used?")
print("\nFollow-up answer:", answer2)