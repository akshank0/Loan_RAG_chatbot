import streamlit as st
from vectorstore import VectorStore
from chatbot import ask_gemini

st.set_page_config(page_title="Loan RAG Chatbot", layout="centered")
st.title("📊 Loan Approval Chatbot")
st.write("Ask any question related to loan approval based on real applicant data.")

query = st.text_input("Enter your question")

if query:
    with st.spinner("Processing..."):
        store = VectorStore()
        context = store.search(query)
        answer = ask_gemini(query, context)

    st.markdown("### 🤖 Answer")
    st.success(answer)

    with st.expander("📄 Retrieved Context"):
        for i, chunk in enumerate(context, 1):
            st.markdown(f"**{i}.** {chunk}")
