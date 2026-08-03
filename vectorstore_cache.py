# Shared application-level cache utilities for Chroma vector store instances.
from typing import Any

import chromadb
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def _collection_name(owner_id: str) -> str:
    """Build a stable Chroma collection name for a logical owner id."""
    safe_owner = "".join(ch for ch in owner_id.strip().lower() if ch.isalnum() or ch == "_")
    if not safe_owner:
        safe_owner = "shared"
    return f"osim_{safe_owner}"


def _as_bool(value: Any, default: bool = False) -> bool:
    """Normalize arbitrary secret values to boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@st.cache_resource(show_spinner=False)
def _get_chroma_client(
    cloud_api_key: str | None,
    cloud_tenant: str | None,
    cloud_database: str | None,
    host: str,
    port: int,
    ssl: bool,
    auth_token: str | None,
) -> Any:
    """Create one shared Chroma client, preferring Chroma Cloud when configured."""
    # Prefer native Chroma Cloud credentials when provided.
    if cloud_api_key and cloud_tenant and cloud_database:
        cloud_client_ctor = getattr(chromadb, "CloudClient", None)
        if cloud_client_ctor is not None:
            return cloud_client_ctor(
                api_key=cloud_api_key,
                tenant=cloud_tenant,
                database=cloud_database,
            )

    # Fallback to plain HTTP endpoint configuration.
    headers = None
    if auth_token:
        headers = {"Authorization": f"Bearer {auth_token}"}
    return chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)


def _connection_config() -> tuple[str | None, str | None, str | None, str, int, bool, str | None]:
    """Read Chroma connection settings from Streamlit secrets."""
    cloud_api_key = st.secrets.get("CHROMA_API_KEY")
    cloud_tenant = st.secrets.get("CHROMA_TENANT")
    cloud_database = st.secrets.get("CHROMA_DATABASE")

    host = str(st.secrets.get("CHROMA_HOST", "localhost"))
    port = int(st.secrets.get("CHROMA_PORT", 8000))
    ssl = _as_bool(st.secrets.get("CHROMA_SSL", False))
    auth_token = st.secrets.get("CHROMA_AUTH_TOKEN")

    return cloud_api_key, cloud_tenant, cloud_database, host, port, ssl, auth_token


@st.cache_resource(show_spinner=False)
def _load_cached_vectorstore(
    owner_id: str,
    collection_name: str,
    api_key: str,
    cloud_api_key: str | None,
    cloud_tenant: str | None,
    cloud_database: str | None,
    host: str,
    port: int,
    ssl: bool,
    auth_token: str | None,
) -> Chroma | None:
    """Return a cached Chroma handle for a remote collection, if it has data."""
    client = _get_chroma_client(
        cloud_api_key,
        cloud_tenant,
        cloud_database,
        host,
        port,
        ssl,
        auth_token,
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    preview = store.get(limit=1)
    if not preview.get("ids"):
        return None
    return store


@st.cache_resource(show_spinner=False)
def _get_or_create_cached_vectorstore(
    owner_id: str,
    collection_name: str,
    api_key: str,
    cloud_api_key: str | None,
    cloud_tenant: str | None,
    cloud_database: str | None,
    host: str,
    port: int,
    ssl: bool,
    auth_token: str | None,
) -> Chroma:
    """Return a cached Chroma handle for write operations."""
    client = _get_chroma_client(
        cloud_api_key,
        cloud_tenant,
        cloud_database,
        host,
        port,
        ssl,
        auth_token,
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def get_cached_vectorstore(owner_id: str) -> Chroma | None:
    """Get the app-cached vector store for a given owner id when collection has data."""
    cloud_api_key, cloud_tenant, cloud_database, host, port, ssl, auth_token = _connection_config()
    return _load_cached_vectorstore(
        owner_id,
        _collection_name(owner_id),
        st.secrets["OPENAI_API_KEY"],
        cloud_api_key,
        cloud_tenant,
        cloud_database,
        host,
        port,
        ssl,
        auth_token,
    )


def get_or_create_cached_vectorstore(owner_id: str) -> Chroma:
    """Get the app-cached vector store for a given owner id for write operations."""
    cloud_api_key, cloud_tenant, cloud_database, host, port, ssl, auth_token = _connection_config()
    return _get_or_create_cached_vectorstore(
        owner_id,
        _collection_name(owner_id),
        st.secrets["OPENAI_API_KEY"],
        cloud_api_key,
        cloud_tenant,
        cloud_database,
        host,
        port,
        ssl,
        auth_token,
    )


def clear_vectorstore_cache() -> None:
    """Clear all cached Chroma handles so future reads reload from remote store."""
    _load_cached_vectorstore.clear()
    _get_or_create_cached_vectorstore.clear()
