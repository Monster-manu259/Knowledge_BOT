import io
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"


def get_env_config() -> tuple[str, str, str]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    return groq_api_key, pinecone_api_key, index_name


def validate_env(groq_api_key: str, pinecone_api_key: str, index_name: str) -> None:
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


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def build_file_signature(file_name: str, pdf_bytes: bytes) -> str:
    return f"{file_name}_{len(pdf_bytes)}"


def extract_texts_from_pdfs(uploaded_files) -> tuple[list[str], str]:
    all_chunks: list[str] = []
    file_ids: list[str] = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()
        file_ids.append(build_file_signature(uploaded_file.name, pdf_bytes))

        text = extract_text_from_pdf(pdf_bytes)
        if not text:
            continue

        chunks = split_text(text)
        all_chunks.extend(
            f"Source: {uploaded_file.name}\n{chunk}" for chunk in chunks
        )

    return all_chunks, "|".join(sorted(file_ids))


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
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL,
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
    file_id: str,
) -> PineconeVectorStore:
    cached_id = st.session_state.get("indexed_file_id")
    cached_store = st.session_state.get("vector_store")

    if cached_store and cached_id == file_id:
        return cached_store

    vector_store = PineconeVectorStore.from_texts(
        chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    st.session_state.indexed_file_id = file_id
    st.session_state.vector_store = vector_store
    return vector_store


def main() -> None:
    st.set_page_config(page_title="THE KNOWLEDGE BOT", layout="wide")
    st.title("📄 THE KNOWLEDGE BOT")

    groq_api_key, pinecone_api_key, index_name = get_env_config()
    validate_env(groq_api_key, pinecone_api_key, index_name)

    uploaded_files = st.file_uploader(
        "Upload PDF",
        type="pdf",
        accept_multiple_files=True,
    )
    if not uploaded_files:
        return

    file_id = "|".join(
        sorted(
            build_file_signature(uploaded_file.name, uploaded_file.getvalue())
            for uploaded_file in uploaded_files
        )
    )

    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully")

    chunks, _ = extract_texts_from_pdfs(uploaded_files)
    if not chunks:
        st.error("No readable text found in the uploaded PDFs.")
        return

    if not chunks:
        st.error("Unable to create text chunks from these PDFs.")
        return

    embeddings = get_embeddings()
    pc = Pinecone(api_key=pinecone_api_key)
    ensure_index(pc, index_name)

    vector_store = get_vector_store(chunks, embeddings, index_name, file_id)

    st.write(f"Total Chunks: {len(chunks)}")
    query = st.text_input("Ask your question")

    if not query:
        return

    with st.spinner("Searching..."):
        docs = vector_store.similarity_search(query, k=TOP_K)
        context = "\n\n".join(doc.page_content for doc in docs)

        llm = get_llm(groq_api_key)
        response = llm.invoke(build_prompt(context, query))
        answer = response.content if hasattr(response, "content") else str(response)

    st.subheader("Answer")
    st.text_area("", value=answer, height=500)


if __name__ == "__main__":
    main()