import PyPDF2
import numpy as np
import faiss
import os
from fastembed import TextEmbedding

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


class PDFChatEngine:
    def __init__(self):
        # Each PDF gets its own index stored here
        # Key: session_id, Value: {"chunks": [...], "index": faiss_index}
        self.sessions = {}
        self.active_session_id = None

    def extract_text(self, pdf_path: str) -> str:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def chunk_text(self, text: str, chunk_size=500, overlap=100) -> list:
        words = text.split()
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def build_index(self, chunks: list):
        embeddings = np.array(list(embedder.embed(chunks))).astype("float32")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def process_pdf(self, pdf_path: str, pdf_name: str, session_id: int) -> int:
        """Process a new PDF and store its index under session_id"""
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)
        index = self.build_index(chunks)

        # Store in memory, keyed by session_id
        self.sessions[session_id] = {
            "chunks": chunks,
            "index": index,
            "pdf_name": pdf_name
        }
        self.active_session_id = session_id
        return len(chunks)

    def switch_session(self, session_id: int) -> bool:
        """Switch active PDF — returns False if not loaded in memory"""
        if session_id in self.sessions:
            self.active_session_id = session_id
            return True
        return False  # PDF needs to be re-uploaded (server was restarted)

    def retrieve(self, query: str, top_k=4) -> list:
        """Search in the currently active PDF's index"""
        if self.active_session_id not in self.sessions:
            return []
        session = self.sessions[self.active_session_id]
        query_vec = np.array(list(embedder.embed([query]))).astype("float32")
        distances, indices = session["index"].search(query_vec, top_k)
        return [session["chunks"][i] for i in indices[0] if i < len(session["chunks"])]

    @property
    def is_ready(self) -> bool:
        return self.active_session_id in self.sessions if self.active_session_id else False