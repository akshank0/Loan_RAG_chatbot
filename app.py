import streamlit as st
from vectorstore import VectorStore
from chatbot import ask_gemini

# Page config
st.set_page_config(page_title="Loan RAG Chatbot", page_icon="💬", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        .title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            color: #2C3E50;
        }
        .subtitle {
            font-size: 1.1rem;
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 20px;
        }
        .stTextInput > div > div > input {
            font-size: 1.1rem;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="title">📊 Loan Approval Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions based on real loan applicant data</div>', unsafe_allow_html=True)

# Input box
query = st.text_input("💬 What would you like to know?")

@st.cache_resource(show_spinner="🔍 Indexing loan data...")
def load_store():
    return VectorStore()

store = load_store()

if query:
    with st.spinner("🔍 Searching through loan data..."):
        context = store.search(query)
        answer = ask_gemini(query, context)

    # Display Answer
    st.markdown("### ✅ Answer")
    st.success(answer)

    # Expandable retrieved context
    with st.expander("📄 Show Retrieved Context", expanded=False):
        for i, chunk in enumerate(context, 1):
            st.markdown(f"**{i}.** {chunk}")
