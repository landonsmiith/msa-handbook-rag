import streamlit as st
from tinyllama_inference import generate_response, retrieve_context

st.set_page_config(page_title="MSA Handbook Assistant", page_icon="📘")
st.markdown("<h1 style='color:#2F4F4F;'>📘 MSA 2025 Handbook Assistant</h1>", unsafe_allow_html=True)
st.write("Ask questions about the MSA 2025 Handbook.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking... generating response..."):
        response = generate_response(query)
        context = retrieve_context(query)

    st.subheader("💬 Answer")
    st.write(response)
