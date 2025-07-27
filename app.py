# app.py
import streamlit as st
from vectorstore import preprocess_data, build_faiss_index
from chatbot import ask_gemini

st.set_page_config(page_title="Loan Assistant", page_icon="💰", layout="wide")

# 🎨 Custom CSS
st.markdown("""
<style>
    .stChatInput {
        font-size: 16px;
    }
    .chat-box {
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 10px;
        background-color: #f0f2f6;
    }
    .chat-question {
        font-weight: bold;
        color: #2c3e50;
    }
    .chat-answer {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# 📊 Header
st.markdown("## 💰 Loan Approval Assistant")
st.markdown("Ask questions based on real loan data and get smart answers using **Gemini AI + FAISS**.")

# 📁 Load and cache
@st.cache_resource
def setup():
    sentences = preprocess_data("data/Training_Dataset.csv")
    index, model = build_faiss_index(sentences)
    return sentences, index, model

sentences, index, model = setup()

# 📌 Sidebar Instructions
with st.sidebar:
    st.markdown("### 📌 Instructions")
    st.markdown("- Ask about patterns like:\n  - _What kind of applicants get approved?_\n  - _Does higher income lead to approval?_\n  - _What is the role of credit history?_")
    st.markdown("- Your questions are matched with real data.\n- Powered by **Gemini API + Embeddings**.")
    st.markdown("---")
    st.markdown("Made with ❤️ by Akshank")

# 💬 Chat Interface
st.markdown("### 🧠 Ask a Question")

question = st.text_input("🔍 Type your question here", placeholder="e.g., Does being self-employed affect loan approval?")
if question:
    with st.spinner("🤖 Thinking..."):
        answer = ask_gemini(question, sentences, index, model)

        # Display Q&A
        st.markdown(f"""<div class='chat-box chat-question'>🧑‍💼 You: {question}</div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class='chat-box chat-answer'>🤖 Gemini: {answer}</div>""", unsafe_allow_html=True)
