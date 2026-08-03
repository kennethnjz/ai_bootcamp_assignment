# Used to print full stack traces when chat calls fail.
import traceback

# Streamlit UI framework for page rendering and session state handling.
import streamlit as st
# OpenAI SDK client for chat completion calls.
from openai import OpenAI
# Chroma vector store class for Chroma Cloud embeddings retrieval.
from langchain_chroma import Chroma
# Authentication and user-scoped vector-store utilities from local module.
from auth import is_logged_in, current_user, logout
# Shared app-level cache helper used by both admin and user pages.
from vectorstore_cache import get_cached_vectorstore


def retrieve_context(vectorstore, query: str, k: int = 4) -> tuple[str, list]:
    """Retrieve top-k relevant chunks from the vector store and join them as a context string."""
    # Build a retriever from the vector store with the requested top-k value.
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # Run retrieval against the query text.
    docs = retriever.invoke(query)
    # Join all chunk contents with separators so the model can read distinct chunks.
    joined_context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    # Return both the merged context text and the original doc objects.
    return joined_context, docs


def build_rag_system_prompt(context: str) -> str:
    """Create a strict RAG system prompt that limits answers to retrieved context only."""
    # Return a system instruction that prevents answers outside the provided context.
    return (
        "Answer ONLY using the context below. "
        'If the answer is not in the context, say "I couldn\'t find that information in the uploaded document."\n\n'
        f"Context:\n{context}"
    )


# Default assistant behavior when no document context is available.
system_prompt_no_doc = "You are a helpful assistant."


def load_persisted_store_for(user_id: str) -> Chroma | None:
    """Load app-cached Chroma Cloud vector store by owner id if present."""
    # Resolve shared app-level cached handle for this owner id.
    return get_cached_vectorstore(user_id)


def load_persisted_user_store() -> Chroma | None:
    """Load the logged-in user's Chroma Cloud vector store if present."""
    # Load the store keyed to the currently logged-in user id.
    return load_persisted_store_for(current_user())


# Initialize default retrieval depth in session state.
if "k_value" not in st.session_state:
    st.session_state.k_value = 4
# Initialize default chat model in session state.
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"
# Initialize default temperature in session state.
if "temperature" not in st.session_state:
    st.session_state.temperature = 1.0

# Create OpenAI API client from Streamlit secret.
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# Read retrieval depth into local variable.
k_value = st.session_state.k_value
# Read active model into local variable.
model = st.session_state.model
# Read active temperature into local variable.
temperature = st.session_state.temperature


# Redirect to main page if no valid logged-in user session exists.
if not is_logged_in() or not current_user().lower().startswith("user"):
    st.switch_page("main.py")

# Render page title.
st.title("User Page")
# Show current logged-in username.
st.write(f"Welcome, **{current_user()}**!")
# Show role-level status message.
st.info("You have standard user access.")

# Initialize chat history container in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Resolve vector store for this request using shared app-level cache.
active_owner = current_user()
# First preference: the logged-in user's own Chroma Cloud collection.
store = load_persisted_user_store()
# Fallback: shared admin-indexed Chroma Cloud collection under the literal "user" scope.
if store is None:
    store = load_persisted_store_for("user")
    if store is not None:
        active_owner = "user"

# Render sequence diagram and graph only when a vector store exists.
if store is not None:
    # Read metadata for UI summary count.
    store_meta = store.get()
    # Confirm vector store availability and indicate source scope.
    if active_owner.lower() == current_user().lower():
        st.success(f"Vector store is available for user '{current_user()}'.")
    else:
        st.success(f"Using shared vector store from '{active_owner}'.")
    # Show count of indexed document IDs from the store metadata.
    st.caption(f"Indexed documents in session store: {len(store_meta.get('ids', []))}")
else:
    # Inform user that no store exists yet for retrieval-backed behavior.
    st.info("No vector store is available in this session yet.")

# Replay full prior chat history in UI.
for message in st.session_state.messages:
    # Render each message in role-appropriate chat bubble.
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Capture new user message from chat input box.
if prompt := st.chat_input("Type a message..."):
    # Persist user message in chat history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Render the just-submitted user message bubble.
    with st.chat_message("user"):
        st.write(prompt)

    # Start assistant response bubble.
    with st.chat_message("assistant"):
        try:
            # Use RAG context only when user-specific store is available.
            if store is not None:
                # Retrieve relevant context and source chunks for this user question.
                context, source_docs = retrieve_context(store, prompt, k_value)
                # Build strict context-bound system prompt.
                effective_system_prompt = build_rag_system_prompt(context)
            else:
                # No store means no sources are available.
                source_docs = []
                # Fall back to a generic system prompt.
                effective_system_prompt = system_prompt_no_doc

            # Build full message list with current system instruction at the front.
            api_messages = [{"role": "system", "content": effective_system_prompt}] + st.session_state.messages
            # Start streaming chat completion from model.
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                stream=True,
            )
            # Stream assistant reply directly into the UI and capture final text.
            reply = st.write_stream(stream)

            # Show retrieval source chunks when RAG context was used.
            if source_docs:
                with st.expander("🔍 View Sources"):
                    # Render each source chunk with an index label.
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.caption(doc.page_content)
        except Exception:
            # Print full traceback to logs for debugging.
            traceback.print_exc()
            # Set fallback reply text for UI continuity.
            reply = "⚠️ Sorry, something went wrong. Please try again."
            # Show visible error state to the user.
            st.error(reply)

    # Persist assistant response in chat history.
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Provide logout action button.
if st.button("Logout"):
    # Clear auth/session state through helper.
    logout()
    # Redirect user to main login page.
    st.switch_page("main.py")
