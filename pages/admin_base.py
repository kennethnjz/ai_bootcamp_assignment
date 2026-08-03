import hashlib          # for generating MD5 hashes to detect row changes
import pandas as pd     # for reading and manipulating Excel data
import streamlit as st  # for building the web UI
from io import BytesIO  # for treating the uploaded file bytes as a file-like object
from langchain_chroma import Chroma                    # vector store backed by ChromaDB
from langchain_core.documents import Document          # LangChain's document wrapper (content + metadata)
from langchain_openai import OpenAIEmbeddings          # OpenAI embedding model to convert text to vectors
from auth import is_logged_in, current_user, logout    # custom auth helpers (your own module)


# @st.cache_resource caches the returned object across all sessions and reruns.
# This means all users share the same in-memory vector store object,
# rather than each user creating their own separate instance.
@st.cache_resource
def get_shared_store():
    # Returns a mutable dict so we can update vectorstore in-place later.
    # A plain None wouldn't be mutable, but a dict wrapper is.
    return {"vectorstore": None}


def _row_hash(text: str) -> str:
    # Generates a short MD5 fingerprint of a row's text content.
    # Used later to detect whether a row has changed between uploads —
    # if the hash is the same, the row hasn't changed and doesn't need re-indexing.
    return hashlib.md5(text.encode()).hexdigest()


def load_and_split(file, system_code: str) -> list[Document]:
    # Read the uploaded Excel file into a pandas DataFrame.
    # BytesIO wraps the raw file bytes so pandas can treat it like a file on disk.
    # engine="openpyxl" explicitly uses the openpyxl library to parse .xlsx files.
    df = pd.read_excel(BytesIO(file.read()), engine="openpyxl", sheet_name="OM")

    # Drop rows where every single column is NaN (completely empty rows),
    # then reset the index so it runs 0, 1, 2, ... cleanly after dropping.
    df = df.dropna(how="all").reset_index(drop=True)

    docs = []  # will hold the final list of LangChain Document objects

    for _, row in df.iterrows():  # iterate over each row; _ discards the row index
        # Concatenate all column-value pairs into a single string, e.g.:
        # "Job Name: DKSD001 | Job Frequency: Daily | Description: FTP receive..."
        # This becomes the text that gets embedded and searched later.
        text = " | ".join(f"{col}: {val}" for col, val in row.items())

        # Extract the job name and clean up whitespace
        job_name = str(row["Job Name"]).strip()

        # Skip rows with no valid job name — these are likely header continuations
        # or empty rows that slipped through the dropna above.
        if not job_name or job_name.lower() == "nan":
            continue

        # Wrap the text in a LangChain Document with metadata.
        # page_content is what gets embedded and retrieved.
        # metadata carries structured fields for filtering, change detection,
        # and scoping deletions to the correct system.
        docs.append(Document(
            page_content=text,
            metadata={
                "job_name": job_name,
                "row_hash": _row_hash(text),  # fingerprint for change detection
                "system_code": system_code     # identifies which system this job belongs to
            }
        ))

    return docs


def get_store_summary(vs: Chroma) -> pd.DataFrame:
    # Fetch all metadata from the store and count jobs per system_code.
    all_metas = vs.get()["metadatas"]
    counts = {}
    for meta in all_metas:
        code = meta.get("system_code", "Unknown")
        counts[code] = counts.get(code, 0) + 1
    return pd.DataFrame(
        [{"System Code": code, "Jobs in Store": count} for code, count in sorted(counts.items())]
    )


def build_vectorstore(docs: list[Document]) -> Chroma:
    # Initialise the OpenAI embedding model.
    # text-embedding-3-small is cost-efficient and accurate enough for this use case.
    # api_key is pulled from Streamlit's secrets manager (secrets.toml), not hardcoded.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"],
    )

    # Create a new in-memory Chroma vector store from the documents.
    # This embeds all documents and stores their vectors in memory.
    # No persist_directory is set, so this lives only for the session lifetime.
    return Chroma.from_documents(docs, embedding=embeddings)


def upsert_documents(vs: Chroma, new_docs: list[Document], system_code: str):
    # Build a dict of {job_name: Document} from the newly uploaded file,
    # so we can look up each job by name efficiently.
    new_by_job = {doc.metadata["job_name"]: doc for doc in new_docs}

    existing_ids, existing_metas = [], []

    # For each job in the new upload, query the store by job_name to find any existing entry.
    for job_name in new_by_job:
        result = vs.get(where={"job_name": job_name})
        existing_ids.extend(result["ids"])         # ChromaDB internal IDs of matched documents
        existing_metas.extend(result["metadatas"]) # metadata dicts of matched documents

    # Build a lookup of {job_name: {id, row_hash}} for jobs already in the store
    # that match names in the incoming file. Used to compare hashes below.
    old_by_job = {
        meta["job_name"]: {"id": doc_id, "row_hash": meta["row_hash"]}
        for doc_id, meta in zip(existing_ids, existing_metas)
    }

    to_delete, to_add = [], []  # accumulate IDs to delete and documents to add

    for job_name, new_doc in new_by_job.items():
        if job_name not in old_by_job:
            # New job not yet in the store — queue for addition
            to_add.append(new_doc)
        elif new_doc.metadata["row_hash"] != old_by_job[job_name]["row_hash"]:
            # Job exists but content has changed — queue old entry for deletion and re-add.
            # ChromaDB has no in-place update, so delete + add is the required pattern.
            to_delete.append(old_by_job[job_name]["id"])
            to_add.append(new_doc)
        # Hash matches — row is unchanged, no action needed

    # Find jobs in the store that belong to this system but are missing from the new upload.
    # Scoping by system_code ensures jobs from other systems are never affected.
    system_existing = vs.get(where={"system_code": system_code})
    system_job_names_in_store = {meta["job_name"] for meta in system_existing["metadatas"]}
    incoming_job_names = set(new_by_job.keys())

    # Jobs present in the store for this system but absent in the new file have been removed
    for job_name in system_job_names_in_store - incoming_job_names:
        result = vs.get(where={"job_name": job_name})
        to_delete.extend(result["ids"])

    if to_delete:
        vs.delete(ids=to_delete)

    if to_add:
        vs.add_documents(to_add)

    return len(to_delete), len(to_add)


# --- Page access control ---
# Redirect non-logged-in users and non-admins back to the main page.
# Any username starting with "admin" is treated as an admin.
if not is_logged_in() or not current_user().lower().startswith("admin"):
    st.switch_page("main.py")


# --- Page UI ---
st.title("Admin Page")
st.write(f"Welcome, **{current_user()}**!")

st.divider()

# Show a live summary of jobs per system currently in the vector store.
# Reads directly from the shared store so it reflects the latest state after every rerun.
shared = get_shared_store()
if shared["vectorstore"] is not None:
    st.subheader("📊 Jobs in Vector Store")
    st.dataframe(get_store_summary(shared["vectorstore"]), hide_index=True, use_container_width=True)
else:
    st.info("No documents indexed yet. Upload an operating manual below to get started.")

st.divider()

# Initialise the clear flag used to reset the system code field after indexing.
# Must be set before the widget renders so the reset happens on the correct rerun.
if "clear_system_code" not in st.session_state:
    st.session_state.clear_system_code = False

# If a previous indexing run set the clear flag, reset the widget value and clear the flag
# before the widget is instantiated — this is the only safe window to modify a widget's key.
if st.session_state.clear_system_code:
    st.session_state.clear_system_code = False
    st.session_state.system_code_input = ""

system_code = st.text_input("🔑 System Code (e.g. DKS)", key="system_code_input").strip().upper()

uploaded_file = st.file_uploader("📄 Upload an Operating Manual (.xlsx)", type=["xlsx"])

if uploaded_file:
    st.success(f"Loaded: {uploaded_file.name}")

    if st.button("Index Document"):
        if not system_code:
            st.error("Please enter a System Code before indexing.")
        else:
            with st.spinner("Indexing..."):
                docs = load_and_split(uploaded_file, system_code)

                if "index_log" not in st.session_state:
                    st.session_state.index_log = []

                if shared["vectorstore"] is None:
                    # First upload — build the vector store from scratch
                    valid_docs = [d for d in docs if d.metadata["job_name"].upper().startswith(system_code)]
                    invalid = len(docs) - len(valid_docs)
                    if not valid_docs:
                        st.error(f"No jobs found with prefix '{system_code}'. Please check the system code and try again.")
                    else:
                        if invalid:
                            st.warning(f"{invalid} job(s) skipped — job name does not start with '{system_code}'.")
                        shared["vectorstore"] = build_vectorstore(valid_docs)
                        msg = f"✅ {uploaded_file.name} ({system_code}) — {len(valid_docs)} jobs added"
                        st.session_state.index_log.append(msg)
                        st.session_state.clear_system_code = True
                        st.rerun()
                else:
                    # Subsequent upload — diff against existing store and apply changes only
                    valid_docs = [d for d in docs if d.metadata["job_name"].upper().startswith(system_code)]
                    invalid = len(docs) - len(valid_docs)
                    if not valid_docs:
                        st.error(f"No jobs found with prefix '{system_code}'. Please check the system code and try again.")
                    else:
                        if invalid:
                            st.warning(f"{invalid} job(s) skipped — job name does not start with '{system_code}'.")
                        deleted, added = upsert_documents(shared["vectorstore"], valid_docs, system_code)
                        msg = f"✅ {uploaded_file.name} ({system_code}) — {added} job(s) added/updated, {deleted} job(s) removed"
                        st.session_state.index_log.append(msg)
                        st.session_state.clear_system_code = True
                        st.rerun()
else:
    st.info("Upload operating manuals to enable jobs analysis Q&A.")

st.divider()

# Display a persistent log of all indexing actions performed this session
if "index_log" in st.session_state and st.session_state.index_log:
    st.subheader("📋 Indexed Documents")
    for entry in st.session_state.index_log:
        st.write(entry)

st.divider()

if st.button("Logout"):
    logout()
    st.switch_page("main.py")