import streamlit as st
import pandas as pd
from langchain_chroma import Chroma

# @st.cache_resource caches the returned object across all sessions and reruns.
# This means all users share the same in-memory vector store object,
# rather than each user creating their own separate instance.
@st.cache_resource
def get_shared_store():
    # Returns a mutable dict so we can update vectorstore in-place later.
    # A plain None wouldn't be mutable, but a dict wrapper is.
    return {"vectorstore": None}

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
