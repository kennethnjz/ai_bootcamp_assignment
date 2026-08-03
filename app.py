import os
import traceback
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

#load_dotenv()
#st.write("OPENAI_API_KEY:", st.secrets["OPENAI_API_KEY"])
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def load_and_split(uploaded_file, chunk_size=800, chunk_overlap=100):
    ext = ".txt" if uploaded_file.name.endswith(".txt") else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        loader = TextLoader(tmp_path, encoding="utf-8") if ext == ".txt" else PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.remove(tmp_path)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)


@st.cache_resource
def build_vectorstore(_chunks):
    return Chroma.from_documents(_chunks, OpenAIEmbeddings(model="text-embedding-3-small"))

def retrieve_context(vectorstore, query, k=4):
    print("calling retrieve_context")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs), docs

def build_rag_system_prompt(context):
    print("calling build_rag_system_prompt")
    return (
        "Answer ONLY using the context below. "
        "If the answer is not in the context, say \"I couldn't find that information in the uploaded document.\"\n\n"
        f"Context:\n{context}"
    )

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Settings")
    
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=4000, value=800, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=100, step=10)

    uploaded_file = st.file_uploader("📄 Upload a Document", type=["txt", "pdf"])
    if uploaded_file:
        st.session_state.pop("doc_summary", None)
        chunks = load_and_split(uploaded_file, chunk_size, chunk_overlap)
        st.session_state.vectorstore = build_vectorstore(chunks)
        st.success(f"Ready! Indexed {len(chunks)} chunks.")

        if "doc_summary" not in st.session_state:
            full_text = " ".join(chunk.page_content for chunk in chunks)[:4000]
            summary_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize the following document in one paragraph."},
                    {"role": "user", "content": full_text},
                ],
            )
            st.session_state.doc_summary = summary_response.choices[0].message.content

        st.info(f"📋 **Summary:** {st.session_state.doc_summary}")
    else:
        st.info("Upload a PDF or TXT to enable document Q&A.")
        system_prompt_no_doc = "You are a helpful assistant."
    st.divider()

    k_value = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)

    PERSONAS = {
        "Helpful Assistant": "You are a helpful assistant.",
        "Singlish Hawker Uncle": "Friendly uncle, Singapore food only, Singlish phrases.",
        "Strict Grammar Teacher": "Corrects grammar before answering.",
    }
    selected_persona = st.selectbox("Persona", list(PERSONAS.keys()))
    system_prompt = PERSONAS[selected_persona]
    st.caption(f"📝 {system_prompt}")

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    chars_placeholder = st.empty()
    download_placeholder = st.empty()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            
            if "vectorstore" in st.session_state:
                context, source_docs = retrieve_context(st.session_state.vectorstore, prompt, k_value)
                effective_system_prompt = build_rag_system_prompt(context)
            else:
                effective_system_prompt = system_prompt_no_doc
            
            api_messages = [{"role": "system", "content": effective_system_prompt}] + st.session_state.messages
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                stream=True,
            )
            reply = st.write_stream(stream)
            
            if "vectorstore" in st.session_state:
                with st.expander("🔍 View Sources"):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.caption(doc.page_content)
            
        except Exception as e:
            traceback.print_exc()
            reply = "⚠️ Sorry, something went wrong. Please try again."
            st.error(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Update sidebar placeholders after messages are finalized
total_chars = sum(len(msg["content"]) for msg in st.session_state.messages)
chars_placeholder.caption(f"💬 Total characters in history: {total_chars}")
conversation_text = "\n".join(
    f"{msg['role'].upper()}: {msg['content']}"
    for msg in st.session_state.messages
)
download_placeholder.download_button(
    label="📥 Download Conversation",
    data=conversation_text,
    file_name="conversation.txt",
    mime="text/plain",
    disabled=len(st.session_state.messages) == 0,
)
