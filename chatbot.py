from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

def ask_gemini(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""Use the following information to answer the question clearly and simply for a loan applicant.

Information:
{context}

Question: {question}
"""
    response = model.generate_content(prompt)
    return response.text.strip()

