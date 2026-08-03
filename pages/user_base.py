import streamlit as st
from auth import is_logged_in, current_user, logout

if not is_logged_in() or not current_user().lower().startswith("user"):
    st.switch_page("main.py")

st.title("User Page")
st.write(f"Welcome, **{current_user()}**!")
st.info("You have standard user access.")

if st.button("Logout"):
    logout()
    st.switch_page("main.py")
