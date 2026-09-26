

# 📄 PDF Chat Agent

**An intelligent RAG-powered document Q&A system built with Flask, FAISS, and Groq LLaMA 3**



---

## ✨ Features

- **RAG Pipeline** — Retrieval-Augmented Generation prevents hallucinations by grounding every answer in your document
- **Semantic Search** — BGE embeddings + FAISS vector index for fast, meaning-aware chunk retrieval
- **Multi-PDF Sessions** — Upload multiple documents and switch between them with full session history
- **Persistent Chat History** — Every Q&A pair saved to SQLite with timestamps; reload any past conversation
- **Free LLM** — Powered by LLaMA 3.3 70B via Groq API (no cost)
- **Modern UI** — Dark-themed responsive interface with drag-and-drop upload and source chunk transparency
- **Source Citations** — See exactly which parts of the PDF were used to generate each answer

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│         HTML + CSS + Vanilla JS (Dark Theme)            │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (Flask REST API)
┌──────────────────────▼──────────────────────────────────┐
│                   Flask Backend                          │
│   /upload  →  /ask  →  /sessions  →  /sessions/:id     │
└───────┬──────────────────────────┬───────────────────────┘
        │                          │
┌───────▼───────┐        ┌─────────▼──────────┐
│  RAG Engine   │        │   SQLite Database   │
│               │        │                     │
│ 1. Extract    │        │  sessions table     │
│    (PyPDF2)   │        │  messages table     │
│               │        │                     │
│ 2. Chunk      │        └─────────────────────┘
│    (overlap)  │
│               │        ┌─────────────────────┐
│ 3. Embed      │        │   Groq Cloud API    │
│    (BGE-small)│        │                     │
│               │        │  LLaMA 3.3 70B      │
│ 4. Index      │        │  (Free tier)        │
│    (FAISS)    │        └─────────────────────┘
│               │
│ 5. Retrieve   │
│    (top-k=4)  │
└───────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-chat-agent.git
cd pdf-chat-agent
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com) → API Keys → Create Key

### 5. Run the application

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📦 Project Structure

```
pdf-chat-agent/
│
├── app.py              # Flask backend — REST API routes
├── rag_engine.py       # RAG logic — chunking, embedding, FAISS indexing
├── database.py         # SQLite helpers — sessions & chat history
│
├── templates/
│   └── index.html      # Frontend UI (single-page)
│
├── static/
│   └── style.css       # Dark theme styling
│
├── uploads/            # Uploaded PDFs stored here
├── chat_history.db     # SQLite database (auto-created)
│
├── .env                # API keys (never commit this)
├── .gitignore
└── requirements.txt
```

---

## 🧠 How RAG Works

| Step | Process | Technology |
|------|---------|-----------|
| 1 | Extract raw text from PDF | PyPDF2 |
| 2 | Split into overlapping chunks (500 words, 100 overlap) | Custom chunker |
| 3 | Convert each chunk to a 384-dim vector | BGE-small-en (fastembed) |
| 4 | Store all vectors in a searchable index | FAISS IndexFlatL2 |
| 5 | Embed user query and find top-4 closest chunks | Cosine/L2 similarity |
| 6 | Send `context + question` to LLM | Groq LLaMA 3.3 70B |
| 7 | Return grounded answer + source chunks | Flask JSON response |

**Why RAG instead of sending the whole PDF?**
A 50-page PDF ≈ 25,000 tokens. RAG retrieves only the 4 most relevant ~500-token chunks, making responses faster, cheaper, and more focused.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET`  | `/` | Serve frontend |
| `POST` | `/upload` | Upload and process a PDF |
| `POST` | `/ask` | Ask a question about the active PDF |
| `GET`  | `/sessions` | List all past sessions |
| `GET`  | `/sessions/<id>/messages` | Load chat history for a session |
| `DELETE` | `/sessions/<id>` | Delete a session and its history |

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend | Flask (Python) | Lightweight, easy to deploy |
| LLM | LLaMA 3.3 70B via Groq | Free, fast, high quality |
| Embeddings | BGE-small-en (fastembed) | No PyTorch needed, ONNX-based |
| Vector DB | FAISS (Facebook AI) | Fast similarity search, CPU-friendly |
| PDF parsing | PyPDF2 | Simple text extraction |
| Database | SQLite | Zero-config, built into Python |
| Frontend | HTML + CSS + Vanilla JS | No framework overhead |

---

## 📋 Requirements

```
flask
groq
PyPDF2
fastembed
faiss-cpu
numpy
python-dotenv
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

[MIT](LICENSE) © 2026

---

<div align="center">
Built with ❤️ using Flask · FAISS · Groq · fastembed
</div>
