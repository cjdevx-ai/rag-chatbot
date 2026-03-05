# ECE THESIS LIBRARY ASSISTANT RAG Chatbot

A high-performance Retrieval-Augmented Generation (RAG) chatbot designed to provide grounded, citation-backed answers from your local research documents. Built with a modern "Cool Night Blue" Streamlit UI, powered by Google Gemini and FAISS.

---

### Project Timeline
- **Started:** March 3, 2026
- **Shipped:** March 6, 2026
- **Deployment:** [ece-library-assistant.streamlit.app](https://ece-library-assistant.streamlit.app)

---

### Features
- **Modern Web UI:** Clean, responsive "Cool Night Blue" theme with glassmorphism effects.
- **Deep Retrieval:** Uses `sentence-transformers` (all-MiniLM-L6-v2) for high-quality semantic search.
- **Fast Vector Store:** Powered by `FAISS` for near-instant document retrieval.
- **Grounded Generation:** Uses `Gemini 2.0 Flash` to generate answers strictly based on retrieved context.
- **Source Citations:** Automatically lists sources with similarity scores for every answer.
- **Session Caching:** Local FAISS index caching for lightning-fast startup on subsequent runs.
- **Multi-turn Chat:** Maintains conversation history for follow-up questions.

---

###  Setup Instructions

#### 1. Clone the Repository
```powershell
git clone <your-repo-url>
cd rag-chatbot
```

#### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

#### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
TOP_K=3
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

#### 5. Prepare your Documents
Place your research papers or notes (`.txt` or `.md` files) into the `docs/` folder.

---

### How to Run

#### **Option A: Web UI (Recommended)**
```powershell
streamlit run main.py
```

#### **Option B: CLI Interface**
```powershell
python cli_main.py
```

---

### Architecture
1. **Document Loader:** Recursively loads and chunks text files with a sliding window.
2. **Embedder:** Converts text chunks into 384-dimensional L2-normalized vectors.
3. **Vector Store:** Manages a FAISS index for exact inner-product similarity search.
4. **Retriever:** Fetches the top-K most relevant chunks for a user query.
5. **Generator:** Augments the Gemini prompt with retrieved context and history to produce the final answer.

---

### Deployment (Streamlit Cloud)
This app is ready for Streamlit Cloud deployment:
1. Push to GitHub.
2. Connect to Streamlit Cloud.
3. Add `GEMINI_API_KEY` in the **Advanced Settings > Secrets** section of your Streamlit dashboard.

---
*Powered by Google Gemini & FAISS*
