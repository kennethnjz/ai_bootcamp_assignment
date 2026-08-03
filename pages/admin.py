# Standard library helper for generating deterministic content hashes per row.
import hashlib
# Pandas is used to read and transform Excel rows before embedding.
import pandas as pd
# Streamlit powers the page UI, widgets, and per-session state.
import streamlit as st
# BytesIO allows treating uploaded in-memory bytes as file-like input for pandas.
from io import BytesIO
# LangChain Chroma wrapper used as the persistent vector store.
from langchain_chroma import Chroma
# LangChain Document object wraps row text and metadata for indexing.
from langchain_core.documents import Document
# Local auth/session helpers and vector-store key/path helpers.
from auth import is_logged_in, current_user, logout
# Shared app-level cache helpers for vector store reuse and invalidation.
from vectorstore_cache import get_cached_vectorstore, get_or_create_cached_vectorstore, clear_vectorstore_cache


# -----------------------------
# Vector Store Load Helpers
# -----------------------------
def load_persisted_user_store() -> Chroma | None:
    """Load the shared user vector store from Chroma Cloud, if it exists."""
    # Return app-cached shared store handle for all sessions.
    return get_cached_vectorstore("user")


def get_session_vectorstore():
    """Return the app-level cached shared store handle."""
    # Return the app-level cached shared store handle.
    return load_persisted_user_store()


# -----------------------------
# Parsing and Document Builders
# -----------------------------
def _row_hash(text: str) -> str:
    """Generate a stable MD5 hash for row content to detect changes."""
    # Encode to bytes and return MD5 hex digest.
    return hashlib.md5(text.encode()).hexdigest()


def load_and_split(file, system_code: str) -> list[Document]:
    """Read one Excel file and convert each valid row into a LangChain Document."""
    # Read OM sheet from uploaded .xlsx bytes.
    df = pd.read_excel(BytesIO(file.read()), engine="openpyxl", sheet_name="OM")

    # Remove fully empty rows and normalize the index after filtering.
    df = df.dropna(how="all").reset_index(drop=True)

    # Accumulator for generated documents.
    docs = []

    # Iterate each row and build searchable text + metadata.
    for _, row in df.iterrows():
        # Flatten all columns into a single searchable string chunk.
        text = " | ".join(f"{col}: {val}" for col, val in row.items())

        # Pull and normalize job name from the required column.
        job_name = str(row["Job Name"]).strip()

        # Skip invalid rows where the job name is missing.
        if not job_name or job_name.lower() == "nan":
            continue

        # Create document with content and metadata used by filters/upserts.
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "job_name": job_name,
                    "row_hash": _row_hash(text),
                    "system_code": system_code,
                },
            )
        )

    # Return all parsed documents from this file.
    return docs


# -----------------------------
# Store Summary and Construction
# -----------------------------
def get_store_summary(vs: Chroma) -> pd.DataFrame:
    """Build a per-system summary table from all vector-store metadata."""
    # Pull all metadatas from the store.
    all_metas = vs.get()["metadatas"]
    # Counter map keyed by system_code.
    counts = {}
    # Count each metadata row into its system bucket.
    for meta in all_metas:
        code = meta.get("system_code", "Unknown")
        counts[code] = counts.get(code, 0) + 1
    # Return sorted table for display.
    return pd.DataFrame(
        [{"System Code": code, "Jobs in Store": count} for code, count in sorted(counts.items())]
    )


def build_vectorstore(docs: list[Document], user_id: str) -> Chroma:
    """Create or append documents into the Chroma Cloud collection for this user scope."""
    store = get_or_create_cached_vectorstore(user_id)
    if docs:
        store.add_documents(docs)
    clear_vectorstore_cache()
    return get_or_create_cached_vectorstore(user_id)


def rebuild_user_vectorstore(docs: list[Document], user_id: str) -> Chroma:
    """Hard reset Chroma Cloud collection and rebuild from provided docs."""
    # Read current app-cached store object if present.
    existing_store = get_or_create_cached_vectorstore(user_id)

    # Best-effort clear of existing collection content before directory removal.
    if existing_store is not None:
        try:
            # Preferred collection-level wipe when supported.
            existing_store.delete_collection()
        except Exception:
            try:
                # Fallback: delete all known IDs manually.
                existing_ids = existing_store.get().get("ids", [])
                if existing_ids:
                    existing_store.delete(ids=existing_ids)
            except Exception:
                # Ignore cleanup failures and continue hard-reset flow.
                pass
    # Clear app-level cache before rebuilding so stale objects are not reused.
    clear_vectorstore_cache()

    # Build fresh collection contents from current valid docs.
    new_store = get_or_create_cached_vectorstore(user_id)
    if docs:
        new_store.add_documents(docs)
    # Invalidate cached handles so future reads use newly persisted data.
    clear_vectorstore_cache()
    # Return rebuilt store handle.
    return get_or_create_cached_vectorstore(user_id)


# -----------------------------
# Incremental Upsert Path
# -----------------------------
def upsert_documents(vs: Chroma, new_docs: list[Document], system_code: str):
    """Incrementally sync incoming docs with existing docs for a given system code."""
    # Map incoming docs by job name for quick lookup.
    new_by_job = {doc.metadata["job_name"]: doc for doc in new_docs}

    # Buffers for existing IDs/metadatas fetched per incoming job.
    existing_ids, existing_metas = [], []

    # Fetch existing store entries matching each incoming job name.
    for job_name in new_by_job:
        result = vs.get(where={"job_name": job_name})
        existing_ids.extend(result["ids"])
        existing_metas.extend(result["metadatas"])

    # Build existing lookup to compare row-hash changes.
    old_by_job = {
        meta["job_name"]: {"id": doc_id, "row_hash": meta["row_hash"]}
        for doc_id, meta in zip(existing_ids, existing_metas)
    }

    # Prepare delete/add action lists.
    to_delete, to_add = [], []

    # Compare incoming docs against existing docs by job name.
    for job_name, new_doc in new_by_job.items():
        if job_name not in old_by_job:
            # New job: add it.
            to_add.append(new_doc)
        elif new_doc.metadata["row_hash"] != old_by_job[job_name]["row_hash"]:
            # Changed job: delete old doc ID and add new doc.
            to_delete.append(old_by_job[job_name]["id"])
            to_add.append(new_doc)
        # Unchanged job: no action.

    # Identify jobs previously in this system but removed from incoming file set.
    system_existing = vs.get(where={"system_code": system_code})
    system_job_names_in_store = {meta["job_name"] for meta in system_existing["metadatas"]}
    incoming_job_names = set(new_by_job.keys())

    # Queue deletions for jobs removed from incoming set.
    for job_name in system_job_names_in_store - incoming_job_names:
        result = vs.get(where={"job_name": job_name})
        to_delete.extend(result["ids"])

    # Execute batched deletions if needed.
    if to_delete:
        vs.delete(ids=to_delete)

    # Execute batched additions if needed.
    if to_add:
        vs.add_documents(to_add)

    # Ensure all sessions reload updated data on next access.
    clear_vectorstore_cache()

    # Return change counts for UI logging.
    return len(to_delete), len(to_add)


# -----------------------------
# Access Control
# -----------------------------
# Reject access when user is not logged in or does not have admin prefix.
if not is_logged_in() or not current_user().lower().startswith("admin"):
    st.switch_page("main.py")


# -----------------------------
# Page Header
# -----------------------------
# Render page title.
st.title("Admin Page")
# Render personalized greeting.
st.write(f"Welcome, **{current_user()}**!")

# Draw section divider.
st.divider()


# -----------------------------
# Existing Store Summary
# -----------------------------
# Load/cached store for this session (and disk if available).
current_store = get_session_vectorstore()
if current_store is not None:
    # Show summary title when store exists.
    st.subheader("📊 Jobs in Vector Store")
    # Render per-system count table.
    st.dataframe(get_store_summary(current_store), hide_index=True, use_container_width=True)
else:
    # Explain there is no indexed content yet.
    st.info("No documents indexed yet. Upload an operating manual below to get started.")

# Draw section divider.
st.divider()


# -----------------------------
# Input State Initialization
# -----------------------------
# Initialize flag that controls when system-code input should be reset.
if "clear_system_code" not in st.session_state:
    st.session_state.clear_system_code = False

# If last run requested an input reset, clear widget value before rendering it.
if st.session_state.clear_system_code:
    st.session_state.clear_system_code = False
    st.session_state.system_code_input = ""


# -----------------------------
# Indexing Inputs
# -----------------------------
# System code input used to filter rows and tag metadata.
system_code = st.text_input("🔑 System Code (e.g. DKS)", key="system_code_input").strip().upper()

# Allow selecting multiple Excel files in one indexing run.
uploaded_files = st.file_uploader(
    "📄 Upload Operating Manual files (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Normalize selected names and create stable files key.
    file_names = sorted(file.name for file in uploaded_files)
    files_key = tuple(file_names)
    # Persist key in session for inspection/debugging.
    st.session_state.files_key = files_key
    # Show load confirmation to the admin.
    st.success(f"Loaded {len(uploaded_files)} file(s): {', '.join(file_names)}")

    # Initialize last-indexed key tracker once.
    if "last_indexed_files_key" not in st.session_state:
        st.session_state.last_indexed_files_key = None

    # Run indexing only when button is pressed.
    if st.button("Index Document"):
        # Enforce required system code.
        if not system_code:
            st.error("Please enter a System Code before indexing.")
        # Avoid repeated indexing for exactly same selected file set in this session.
        elif st.session_state.last_indexed_files_key == files_key:
            st.info("These files are already indexed for this session. Upload a different set to re-index.")
        else:
            # Show spinner during parsing/embedding/store operations.
            with st.spinner("Indexing..."):
                # Merge chunks from all selected files into one list.
                docs = []
                for uploaded_file in uploaded_files:
                    docs.extend(load_and_split(uploaded_file, system_code))

                # Initialize index history log list once.
                if "index_log" not in st.session_state:
                    st.session_state.index_log = []

                # Filter docs by system code prefix in job name.
                valid_docs = [d for d in docs if d.metadata["job_name"].upper().startswith(system_code)]
                # Count skipped rows that do not match system code prefix.
                invalid = len(docs) - len(valid_docs)

                # Stop early when nothing valid is left to index.
                if not valid_docs:
                    st.error(f"No jobs found with prefix '{system_code}'. Please check the system code and try again.")
                else:
                    # Warn for skipped rows outside current system prefix.
                    if invalid:
                        st.warning(f"{invalid} job(s) skipped — job name does not start with '{system_code}'.")

                    # First-time store creation when no existing store is loaded.
                    if current_store is None:
                        build_vectorstore(valid_docs, "user")
                        clear_vectorstore_cache()
                        msg = f"✅ {len(uploaded_files)} file(s) ({system_code}) — {len(valid_docs)} jobs added"
                    else:
                        # Check whether this system code already exists in current store.
                        existing_for_system = current_store.get(where={"system_code": system_code})
                        if existing_for_system.get("ids"):
                            # Existing system code: hard reset and rebuild to avoid duplicate accumulation.
                            rebuild_user_vectorstore(valid_docs, "user")
                            msg = (
                                f"✅ {len(uploaded_files)} file(s) ({system_code}) — "
                                f"system code already existed, store was reset and rebuilt with {len(valid_docs)} jobs"
                            )
                        else:
                            # New system code in an existing store: incremental upsert path.
                            deleted, added = upsert_documents(current_store, valid_docs, system_code)
                            msg = (
                                f"✅ {len(uploaded_files)} file(s) ({system_code}) — "
                                f"{added} job(s) added/updated, {deleted} job(s) removed"
                            )

                    # Append result message to history log.
                    st.session_state.index_log.append(msg)
                    # Track last indexed file set.
                    st.session_state.last_indexed_files_key = files_key
                    # Request system code input clear on next rerun.
                    st.session_state.clear_system_code = True
                    # Trigger rerun so UI reflects latest store state.
                    st.rerun()
else:
    # Prompt user to upload files before indexing.
    st.info("Upload operating manuals to enable jobs analysis Q&A.")

# Draw section divider.
st.divider()


# -----------------------------
# Indexing History
# -----------------------------
# Show indexing history if it exists and contains entries.
if "index_log" in st.session_state and st.session_state.index_log:
    st.subheader("📋 Indexed Documents")
    for entry in st.session_state.index_log:
        st.write(entry)

# Draw section divider.
st.divider()


# -----------------------------
# Logout Action
# -----------------------------
# Render logout button and clear auth session when clicked.
if st.button("Logout"):
    logout()
    st.switch_page("main.py")
