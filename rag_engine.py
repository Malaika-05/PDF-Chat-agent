import PyPDF2
import numpy as np
import faiss
import os
import threading
from fastembed import TextEmbedding

# Loaded once, reused across requests
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
_embed_lock = threading.Lock()  # fastembed's onnxruntime session isn't guaranteed thread-safe


def _embed(texts: list) -> np.ndarray:
    """Thread-safe wrapper around embedder.embed()."""
    with _embed_lock:
        return np.array(list(embedder.embed(texts))).astype("float32")


class PDFChatEngine:
    def __init__(self):
        # Each PDF gets its own index stored here
        # Key: session_id, Value: {"chunks": [...], "index": faiss_index, "pdf_name": str}
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

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50,
                   min_chunk_words: int = 40) -> list:
        """
        Word-window chunking with overlap.

        For long documents this behaves like a normal fixed-size sliding
        window (chunk_size=300, overlap=50). For SHORT documents — where
        total word count <= chunk_size — a fixed window always produces
        exactly 1 chunk, which is what was happening with short PDFs like
        single-page scripts. So when the doc is short, we scale chunk_size
        down (never below min_chunk_words) to aim for ~3-4 chunks instead.
        """
        words = text.split()
        total = len(words)
        if total == 0:
            return []

        effective_chunk_size = chunk_size
        effective_overlap = overlap

        if total <= chunk_size:
            target_chunks = max(1, min(4, total // min_chunk_words))
            if target_chunks > 1:
                effective_chunk_size = max(min_chunk_words, total // target_chunks)
                effective_overlap = min(overlap, effective_chunk_size // 4)
            else:
                # Too little text even for 2 chunks of min_chunk_words each
                effective_chunk_size = total
                effective_overlap = 0

        print(f"[chunk_text] total_words={total} -> "
              f"chunk_size={effective_chunk_size}, overlap={effective_overlap}")

        step = max(effective_chunk_size - effective_overlap, 1)
        chunks = []
        seen = set()
        for i in range(0, total, step):
            chunk = " ".join(words[i: i + effective_chunk_size])
            chunk = chunk.strip()
            if len(chunk) > 20 and chunk not in seen:
                chunks.append(chunk)
                seen.add(chunk)

        print(f"[chunk_text] produced {len(chunks)} chunk(s)")
        return chunks

    def build_index(self, chunks: list):
        embeddings = _embed(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def process_pdf(self, pdf_path: str, pdf_name: str, session_id: int) -> int:
        """Process a new PDF and store its index under session_id."""
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)

        if not chunks:
            # Common cause: scanned/image-only PDF with no text layer (needs OCR)
            raise ValueError(
                f"No extractable text found in '{pdf_name}'. "
                "It may be a scanned/image-only PDF that needs OCR."
            )

        index = self.build_index(chunks)

        self.sessions[session_id] = {
            "chunks": chunks,
            "index": index,
            "pdf_name": pdf_name,
        }
        self.active_session_id = session_id
        return len(chunks)

    def switch_session(self, session_id: int) -> bool:
        """Switch active PDF — returns False if not loaded in memory
        (server was restarted, or PDF needs to be re-uploaded)."""
        if session_id in self.sessions:
            self.active_session_id = session_id
            return True
        return False

    def retrieve(self, query: str, top_k: int = 4) -> list:
        """Search in the currently active PDF's index."""
        if self.active_session_id not in self.sessions:
            return []

        session = self.sessions[self.active_session_id]
        n_chunks = len(session["chunks"])
        if n_chunks == 0:
            return []

        # Never ask FAISS for more neighbors than exist — this is what was
        # causing the "-1" padding indices to wrap around to chunks[-1]
        # and get duplicated in the results.
        k = min(top_k, n_chunks)

        query_vec = _embed([query])
        distances, indices = session["index"].search(query_vec, k)

        results = []
        for i in indices[0]:
            if i == -1:
                continue  # FAISS padding value, not a real match
            if 0 <= i < n_chunks:
                results.append(session["chunks"][i])
        return results

    @property
    def is_ready(self) -> bool:
        return self.active_session_id in self.sessions if self.active_session_id else False