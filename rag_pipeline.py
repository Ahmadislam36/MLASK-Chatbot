import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# STEP 0: LOAD ENV FIRST
# =========================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")

genai.configure(api_key=GOOGLE_API_KEY)

# =========================
# STEP 1: LOAD DATA
# =========================


loader = PyPDFLoader("data/ML.pdf")
documents = loader.load()

# =========================
# STEP 2: SPLIT TEXT
# =========================
splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)
docs = splitter.split_documents(documents)

# =========================
# STEP 3: EMBEDDINGS (CACHED)
# =========================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# =========================
# STEP 4: VECTOR DB (CACHED)
# =========================
@st.cache_resource
def load_db():
    embeddings = load_embeddings()

    if os.path.exists("vectorstore"):
        db = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("vectorstore")

    return db

db = load_db()

# =========================
# STEP 5: GEMINI MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    return genai.GenerativeModel("gemini-flash-lite-latest")

model = load_model()

# =========================
# STEP 6: CALL GEMINI
# =========================
def call_gemini(prompt):
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7
            }
        )

        if hasattr(response, "text") and response.text:
            return response.text
        elif response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return "⚠️ No response from Gemini"

    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# =========================
# STEP 7: MAIN FUNCTION
# =========================
def ask_question(query):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in results])

    # ✅ FIXED PROMPT (properly inside function)
    prompt = f"""
You are an AI assistant that helps users understand Machine Learning concepts.

Use the context below to answer the question.

Context:
{context}

Question:
{query}

Instructions:
- Explain in simple and easy language
- Be clear and to the point
- Use examples if helpful
- If answer is not in context, say "I don't know based on the given data"
- Do NOT act like a doctor or medical assistant
- Do NOT add unnecessary information
"""

    answer = call_gemini(prompt)
    return answer, context

# =========================
# TEST RUN
# =========================
if __name__ == "__main__":
    user_query = "What are the symptoms of a common cold?"
    ans, ctx = ask_question(user_query)

    print("\n--- ANSWER ---")
    print(ans)