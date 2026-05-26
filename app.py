import io
import hashlib
import os

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pydantic import SecretStr
from pypdf import PdfReader

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

def get_env_config() -> tuple[str | None, str | None, str | None]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    return groq_api_key, pinecone_api_key, index_name


def validate_env(
    groq_api_key: str | None,
    pinecone_api_key: str | None,
    index_name: str | None,
) -> tuple[str, str, str]:
    missing = []
    if not groq_api_key:
        missing.append("GROQ_API_KEY")
    if not pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if not index_name:
        missing.append("PINECONE_INDEX_NAME")

    if missing:
        st.error("Missing required environment variables: " + ", ".join(missing))
        st.stop()

    return groq_api_key or "", pinecone_api_key or "", index_name or ""


def initialize_session_state() -> None:
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("uploaded_pdf_hashes", set())
    st.session_state.setdefault("processed_selection_signature", "")
    st.session_state.setdefault("chat_history", [])


def hash_pdf_bytes(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_texts_from_pdfs(uploaded_files, known_hashes: set[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    all_chunks: list[str] = []
    new_hashes: list[str] = []
    duplicate_files: list[str] = []
    unreadable_files: list[str] = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()
        pdf_hash = hash_pdf_bytes(pdf_bytes)

        if pdf_hash in known_hashes:
            duplicate_files.append(uploaded_file.name)
            continue

        new_hashes.append(pdf_hash)

        text = extract_text_from_pdf(pdf_bytes)
        if not text:
            unreadable_files.append(uploaded_file.name)
            continue

        chunks = split_text(text)
        all_chunks.extend(
            f"Source: {uploaded_file.name}\n{chunk}" for chunk in chunks
        )

    return all_chunks, new_hashes, duplicate_files, unreadable_files


@st.cache_data(show_spinner=False)
def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_llm(groq_api_key: str) -> ChatGroq:
    return ChatGroq(
        api_key=SecretStr(groq_api_key),
        model=GROQ_MODEL,
        temperature=0.3,
    )


def ensure_index(pc: Pinecone, index_name: str) -> None:
    listed = pc.list_indexes()
    if hasattr(listed, "names"):
        names = set(listed.names())
    else:
        names = {
            item.get("name")
            for item in listed
            if isinstance(item, dict) and item.get("name")
        }

    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def get_existing_vector_count(pc: Pinecone, index_name: str) -> int:
    try:
        stats = pc.Index(index_name).describe_index_stats()
        total = getattr(stats, "total_vector_count", 0)
        return int(total or 0)
    except Exception:
        return 0


def build_prompt(context: str, question: str) -> str:
    return (
        "You are a helpful assistant answering questions from a PDF. "
        "Use only the context below. If the answer is not in the context, "
        "say you don't know.\n\n"
        "You must provide a clear explanation of the answer, not just a quote from the context."
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def get_vector_store(
    chunks: list[str],
    embeddings: HuggingFaceEmbeddings,
    index_name: str,
) -> PineconeVectorStore:
    cached_store = st.session_state.get("vector_store")

    if cached_store:
        cached_store.add_texts(chunks)
        return cached_store

    vector_store = PineconeVectorStore.from_texts(
        chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    st.session_state.vector_store = vector_store
    return vector_store


def get_connected_vector_store(
    embeddings: HuggingFaceEmbeddings,
    index_name: str,
) -> PineconeVectorStore:
    cached_store = st.session_state.get("vector_store")
    if cached_store:
        return cached_store

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )
    st.session_state.vector_store = vector_store
    return vector_store


def render_chat_history() -> None:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main() -> None:
    st.set_page_config(page_title="THE KNOWLEDGE BOT", layout="wide")
    st.title("📄 THE KNOWLEDGE BOT")
    initialize_session_state()

    groq_api_key, pinecone_api_key, index_name = get_env_config()
    groq_api_key, pinecone_api_key, index_name = validate_env(
        groq_api_key,
        pinecone_api_key,
        index_name,
    )

    embeddings = get_embeddings()
    pc = Pinecone(api_key=pinecone_api_key)
    ensure_index(pc, index_name)
    vector_store = get_connected_vector_store(embeddings, index_name)
    existing_vectors = get_existing_vector_count(pc, index_name)

    with st.sidebar:
        st.header("Knowledge Upload")
        st.caption("Upload once, then chat directly.")

        uploaded_files = st.file_uploader(
            "Upload PDF",
            type="pdf",
            accept_multiple_files=True,
        )

        if st.button("Process Uploads", use_container_width=True):
            if not uploaded_files:
                st.info("Select one or more PDFs first.")
            else:
                current_selection_signature = "|".join(
                    sorted(
                        hash_pdf_bytes(uploaded_file.getvalue())
                        for uploaded_file in uploaded_files
                    )
                )

                if current_selection_signature == st.session_state.processed_selection_signature:
                    st.info("These files were already processed in this session.")
                else:
                    chunks, new_hashes, duplicate_files, unreadable_files = extract_texts_from_pdfs(
                        uploaded_files,
                        st.session_state.uploaded_pdf_hashes,
                    )

                    if duplicate_files:
                        st.warning(
                            "Already loaded in this session: " + ", ".join(duplicate_files)
                        )

                    if unreadable_files:
                        st.error(
                            "No readable text found in: " + ", ".join(unreadable_files)
                        )

                    if chunks:
                        vector_store = get_vector_store(chunks, embeddings, index_name)
                        st.session_state.uploaded_pdf_hashes.update(new_hashes)
                        st.success(f"Added {len(new_hashes)} new PDF(s).")
                    else:
                        st.info("No new readable text was added.")

                    st.session_state.processed_selection_signature = current_selection_signature

        st.divider()
        st.write(f"Session uploads: {len(st.session_state.uploaded_pdf_hashes)}")

    st.subheader("Chat with Your Knowledge Base")

    if existing_vectors == 0 and not st.session_state.uploaded_pdf_hashes:
        st.info("Upload PDFs from the sidebar to start chatting.")
        return

    render_chat_history()
    query = st.chat_input("Ask a question about your PDFs...")

    if not query:
        return

    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            docs = vector_store.similarity_search(query, k=TOP_K)
            context = "\n\n".join(doc.page_content for doc in docs)

            llm = get_llm(groq_api_key)
            response = llm.invoke(build_prompt(context, query))
            answer = response.content if hasattr(response, "content") else str(response)

        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()