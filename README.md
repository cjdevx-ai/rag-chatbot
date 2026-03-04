# **RAG Chatbot Step by Step Setup**
AUTHOR: Clarence Jay Fetalino

## Setup

## 1. Clone this Repository (make sure to have git installed)

Open **Command Prompt (cmd)** and run:

```cmd
git clone https://github.com/cjdevx-ai/rag-chatbot.git
```

---

## 2. Create the virtual environment

In **cmd or PowerShell**:

```bash
python -m venv .venv
```

---

## 3. Activate the virtual environment

In **PowerShell**:

```powershell
.venv\Scripts\Activate
```

In **Command Prompt (cmd)**:

```cmd
.venv\Scripts\activate.bat
```

---

## 4. Install packages

```bash
pip install -r requirements.txt
```

---

## Your project should look like this:
Your project should now look like this:
```
rag-chatbot/
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   └── pipeline.py
│
├── docs/                ← your knowledge base .txt files go here
├── main.py              ← Streamlit app
├── .env                 ← your Gemini API key
└── requirements.txt
```

---
