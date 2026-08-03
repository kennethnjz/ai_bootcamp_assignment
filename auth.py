from pathlib import Path
import bcrypt
import streamlit as st


def verify_login(user_id: str, password: str) -> bool:
    users = st.secrets.get("passwords", {})
    key = user_id.lower()
    if key not in users:
        return False
    return bcrypt.checkpw(password.encode(), users[key].encode())


def login(user_id: str):
    st.session_state["user_id"] = user_id


def get_vectorstore_key(user_id: str | None = None) -> str:
    uid = (user_id or current_user() or "shared").strip().lower()
    return f"vectorstore_{uid}" if uid else "vectorstore_shared"


def get_vectorstore_dir(user_id: str | None = None) -> str:
    uid = (user_id or current_user() or "shared").strip().lower()
    directory = Path(__file__).resolve().parent / "vectorstores" / uid
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def logout():
    st.session_state.pop("user_id", None)
    st.session_state.pop("messages", None)


def is_logged_in() -> bool:
    return "user_id" in st.session_state


def current_user() -> str:
    return st.session_state.get("user_id", "")
