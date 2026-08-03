import streamlit as st
from auth import verify_login, login, is_logged_in, current_user

st.title("OSIM App")

if is_logged_in():
    uid = current_user()
    if uid.lower().startswith("admin"):
        st.switch_page("pages/admin.py")
    elif uid.lower().startswith("user"):
        st.switch_page("pages/user.py")

st.subheader("Login")
user_id  = st.text_input("User ID")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if not user_id or not password:
        st.warning("Please enter both User ID and password.")
    elif verify_login(user_id, password):
        login(user_id)
        st.rerun()
    else:
        st.error("Invalid User ID or password.")
