import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from rag_engine import PDFChatEngine

load_dotenv()  # reads .env into os.environ for local runs; no-op on Streamlit Cloud (no .env there)

st.set_page_config(page_title="PDFChat", page_icon="📄", layout="wide")


# ── API key: works both locally (.env) and on Streamlit Community Cloud (secrets) ──
def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # no secrets.toml present (e.g. running locally without one)
    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_api_key()
if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not set.\n\n"
        "- On Streamlit Community Cloud: App settings → Secrets → add `GROQ_API_KEY = \"...\"`\n"
        "- Locally: put `GROQ_API_KEY=...` in a `.env` file"
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ── Per-browser-session state (Streamlit isolates this per user automatically) ──
if "engine" not in st.session_state:
    st.session_state.engine = PDFChatEngine()
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}  # pdf_name -> [{"role", "content", "sources"}]

engine = st.session_state.engine


# ── Sidebar: upload + document switcher ──
with st.sidebar:
    st.title("📄 PDFChat")
    st.caption("RAG-powered document Q&A")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None and st.button("Upload & Process PDF", use_container_width=True):
        os.makedirs("uploads", exist_ok=True)
        pdf_path = os.path.join("uploads", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing PDF..."):
            try:
                session_id = uploaded_file.name  # filename doubles as the session key
                chunk_count = engine.process_pdf(pdf_path, uploaded_file.name, session_id)
                st.session_state.active_doc = session_id
                st.session_state.chat_histories.setdefault(session_id, [])
                st.success(f"Indexed {chunk_count} chunk(s) from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")

    st.divider()
    st.subheader("Documents this session")
    if engine.sessions:
        names = list(engine.sessions.keys())
        default_idx = names.index(st.session_state.active_doc) if st.session_state.active_doc in names else 0
        selected = st.radio("Pick a document", names, index=default_idx, label_visibility="collapsed")
        if selected != st.session_state.active_doc:
            engine.switch_session(selected)
            st.session_state.active_doc = selected

        if st.button("🗑️ Delete this document", use_container_width=True):
            engine.sessions.pop(st.session_state.active_doc, None)
            st.session_state.chat_histories.pop(st.session_state.active_doc, None)
            st.session_state.active_doc = None
            st.rerun()
    else:
        st.caption("No documents uploaded yet.")

    st.divider()
    st.caption(
        "⚠️ This runs in memory for your browser session only. "
        "Refreshing the app or an app restart clears uploaded PDFs — re-upload if that happens."
    )


# ── Main chat area ──
active_doc = st.session_state.active_doc

if not active_doc:
    st.info("Upload a PDF from the sidebar to get started.")
    st.stop()

st.header(active_doc)

history = st.session_state.chat_histories.setdefault(active_doc, [])

for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"{len(msg['sources'])} source chunk(s) used"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:** {chunk}")

question = st.chat_input(f'Ask anything about "{active_doc}"...')
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        relevant_chunks = []
        with st.spinner("Thinking..."):
            try:
                relevant_chunks = engine.retrieve(question, top_k=4)
                context = "\n\n---\n\n".join(relevant_chunks)

                system_prompt = (
                    "You are a precise document assistant.\n"
                    "Answer questions ONLY based on the provided context from the PDF.\n"
                    'If the answer is not in the context, say: "I couldn\'t find this in the document."\n'
                    "Be concise and clear."
                )
                user_prompt = f"Context from PDF:\n{context}\n\nQuestion: {question}"

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.markdown(answer)
        if relevant_chunks:
            with st.expander(f"{len(relevant_chunks)} source chunk(s) used"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Chunk {i}:** {chunk}")

    history.append({"role": "assistant", "content": answer, "sources": relevant_chunks})