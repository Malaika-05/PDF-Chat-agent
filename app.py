import os
from flask import Flask, request, jsonify, render_template
from groq import Groq
from rag_engine import PDFChatEngine
from database import init_db, create_session, save_message, get_sessions, get_messages, delete_session
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs("uploads", exist_ok=True)

# Initialize DB and engine
init_db()
engine = PDFChatEngine()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.route("/")
def index():
    return render_template("index.html")


# ── Get all past sessions ────────────────────────────────────────────────────
@app.route("/sessions", methods=["GET"])
def list_sessions():
    return jsonify(get_sessions())


# ── Load a past session's messages ──────────────────────────────────────────
@app.route("/sessions/<int:session_id>/messages", methods=["GET"])
def load_messages(session_id):
    messages = get_messages(session_id)
    # Try to switch engine to this session
    loaded = engine.switch_session(session_id)
    return jsonify({"messages": messages, "loaded": loaded})


# ── Delete a session ─────────────────────────────────────────────────────────
@app.route("/sessions/<int:session_id>", methods=["DELETE"])
def remove_session(session_id):
    delete_session(session_id)
    if engine.active_session_id == session_id:
        engine.active_session_id = None
    return jsonify({"success": True})


# ── Upload and process PDF ───────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(pdf_path)

    try:
        # Create DB session first to get session_id
        temp_chunks = 0
        session_id = create_session(file.filename, 0)

        # Process PDF with session_id
        chunk_count = engine.process_pdf(pdf_path, file.filename, session_id)

        # Update chunk count in DB
        import sqlite3
        conn = sqlite3.connect("chat_history.db")
        conn.execute("UPDATE sessions SET chunk_count=? WHERE id=?", (chunk_count, session_id))
        conn.commit()
        conn.close()

        return jsonify({
            "message": f"✅ PDF processed successfully!",
            "filename": file.filename,
            "chunks": chunk_count,
            "session_id": session_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Ask a question ───────────────────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id")

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    if not engine.is_ready:
        return jsonify({"error": "Please upload a PDF first"}), 400

    try:
        relevant_chunks = engine.retrieve(question, top_k=4)
        context = "\n\n---\n\n".join(relevant_chunks)

        system_prompt = """You are a precise document assistant.
Answer questions ONLY based on the provided context from the PDF.
If the answer is not in the context, say: "I couldn't find this in the document."
Be concise and clear."""

        user_prompt = f"Context from PDF:\n{context}\n\nQuestion: {question}"

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        # Save to DB
        if session_id:
            save_message(session_id, "user", question)
            save_message(session_id, "assistant", answer, relevant_chunks)

        return jsonify({"answer": answer, "sources": relevant_chunks})

    except Exception as e:
        print(f"ERROR in /ask: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)