import streamlit as st
from rag_pipeline import ask_question

st.title("MLASK Chatbot")

query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
    
        answer, context = ask_question(query)

    st.subheader("Answer:")
    st.write(answer)

