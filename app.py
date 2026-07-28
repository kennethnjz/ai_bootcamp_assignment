import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# Load from .env file if present (create one with: OPENAI_API_KEY=sk-...)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
)

st.title("AI Chatbot")


def load_and_split(uploaded_file, chunk_size=800, chunk_overlap=100):
    file_name = uploaded_file.name.lower()
    extension = os.path.splitext(file_name)[1] or ".txt"
    suffix = extension if extension in {".pdf", ".txt"} else ".txt"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")

        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(documents)
    finally:
        os.remove(temp_path)


@st.cache_resource
def build_vectorstore(_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"])
    return Chroma.from_documents(_chunks, embedding=embeddings, persist_directory=None)


def retrieve_context(vectorstore, query, k=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    documents = retriever.invoke(query)
    joined_context = "\n\n---\n\n".join(doc.page_content for doc in documents)
    return joined_context, documents

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant."

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o-mini"

if "temperature" not in st.session_state:
    st.session_state.temperature = 1.0

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "files_key" not in st.session_state:
    st.session_state.files_key = ()

chunk_size = 800
chunk_overlap = 100

with st.sidebar:
    uploaded_files = st.file_uploader(
        "📄 Upload a Document",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        current_files_key = tuple(sorted(file.name for file in uploaded_files))
        if current_files_key != st.session_state.files_key:
            all_chunks = []
            for uploaded_file in uploaded_files:
                all_chunks.extend(
                    load_and_split(uploaded_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                )
            st.session_state.vectorstore = build_vectorstore(all_chunks)
            st.session_state.files_key = current_files_key

        st.success(f"Ready! Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")
        for uploaded_file in uploaded_files:
            ext = os.path.splitext(uploaded_file.name)[1] or ""
            st.write(f"✓ {uploaded_file.name}{ext}")
    else:
        st.info("Upload a PDF or TXT to enable document Q&A.")

    chunk_size = st.number_input(
        "Chunk size",
        min_value=100,
        value=800,
        step=50,
    )
    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        value=100,
        step=10,
    )
    k_value = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=10,
        value=4,
    )

    st.header("⚙️ Settings")
    persona_options = {
        "Helpful Assistant": "You are a helpful assistant.",
        "Singlish Hawker Uncle": "You are a friendly hawker uncle who only talks about Singapore food in Singlish.",
        "Strict Grammar Teacher": "You correct every grammar mistake before answering.",
    }

    selected_persona = st.selectbox(
        "Persona",
        options=list(persona_options.keys()),
        index=list(persona_options.keys()).index(
            next(
                (name for name, prompt in persona_options.items() if prompt == st.session_state.system_prompt),
                "Helpful Assistant",
            )
        ),
    )
    st.session_state.system_prompt = persona_options[selected_persona]

    st.session_state.selected_model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=["gpt-4o-mini", "gpt-4o"].index(st.session_state.selected_model),
    )

    st.session_state.selected_language = st.selectbox(
        "Reply Language",
        ["English", "Malay", "Chinese", "Tamil"],
        index=["English", "Malay", "Chinese", "Tamil"].index(st.session_state.selected_language),
    )

    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(st.session_state.temperature),
        step=0.1,
    )

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    total_chars = sum(len(msg["content"]) for msg in st.session_state.messages)
    st.caption(f"Conversation characters: {total_chars}")

    chat_text = "\n".join(
        f"[{message['role']}]: {message['content']}" for message in st.session_state.messages
    )
    st.download_button(
        label="📥 Download Chat",
        data=chat_text,
        file_name="conversation.txt",
        mime="text/plain",
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def build_rag_system_prompt(context, persona_prompt):
    return (
        f"{persona_prompt} "
        "You answer ONLY from the provided context. "
        'If the answer is not in the context, say: '
        '"I couldn\'t find that information in the uploaded document."'
        f"\n\nContext:\n{context}"
    )

system_prompt_no_doc = st.session_state.system_prompt

prompt = st.chat_input("Type your message here...")

if prompt is not None:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    collected_chunks = []

    base_prompt = st.session_state.system_prompt

    retrieved_documents = []

    if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
        context, retrieved_documents = retrieve_context(st.session_state.vectorstore, prompt, k_value)
        base_prompt = build_rag_system_prompt(context, base_prompt)
    else:
        base_prompt = system_prompt_no_doc

    system_prompt = f"{base_prompt} Always reply in {st.session_state.selected_language}."
    api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    try:
        def stream_response():
            for chunk in client.chat.completions.create(
                model=st.session_state.selected_model,
                messages=api_messages,
                temperature=st.session_state.temperature,
                stream=True,
            ):
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    collected_chunks.append(delta)
                    yield delta

        with st.chat_message("assistant"):
            st.write_stream(stream_response())

        assistant_reply = "".join(collected_chunks)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_reply}
        )

        if retrieved_documents:
            with st.expander("🔍 View Sources"):
                for idx, doc in enumerate(retrieved_documents, start=1):
                    st.markdown(f"**Chunk {idx}**")
                    st.text(doc.page_content)
    except Exception as e:
        print(f"OpenAI API error: {e}")
        st.error("Sorry, I couldn't complete that request right now.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "Sorry, I couldn't complete that request right now."}
        )

