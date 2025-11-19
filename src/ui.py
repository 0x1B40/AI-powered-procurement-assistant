from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8000/chat"


st.set_page_config(page_title="Procurement Assistant", page_icon="📊")
st.title("California Procurement Assistant")
st.caption("Ask conversational questions; the assistant runs MongoDB queries on the DGS dataset.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Backend Settings")
    api_url = st.text_input("API URL", value=API_URL)
    if st.button("Clear chat history"):
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about purchase orders...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        try:
            response = requests.post(api_url, json={"question": prompt}, timeout=120)
            response.raise_for_status()
            answer = response.json()["answer"]
        except Exception as exc:  # pragma: no cover
            answer = f"Error: {exc}"
        placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

