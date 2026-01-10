import os
import streamlit as st
import re
import numpy as np
import faiss
import pickle
import ollama
from pypdf import PdfReader

from main import (
    extract_text,
    clean_text,
    chunk_text,
    chunk_to_embedding,
    build_faiss_index,
    save_index,
    load_index,
    faiss_search,
    question_embedding,
    generate_answer
)


st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



#header
st.title("Chat with your PDF")
st.caption("Local • Private • Powered by Ollama")

st.divider()

#sidebar
with st.sidebar:
    st.header("Upload PDF")

    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to start asking questions"
    )

    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()


#state init
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False

#pdf processing
if pdf_file and not st.session_state.index_loaded:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    with st.spinner("Indexing PDF..."):
        pages = extract_text(pdf_path)

        if isinstance(pages, str):
            st.error(pages)
        else:
            cleaned_pages = clean_text(pages)
            chunks = chunk_text(cleaned_pages)

            if os.path.exists("index.faiss") and os.path.exists("metadata.pkl"):
                index, metadata = load_index()
            else:
                embedding_store = chunk_to_embedding(chunks)
                index, metadata = build_faiss_index(embedding_store)
                save_index(index, metadata)

            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.index_loaded = True

    st.success("PDF indexed successfully!")

#question area
st.subheader("Ask a question")

if not st.session_state.index_loaded:
    st.info("Upload a PDF to start chatting.")
else:
    question = st.chat_input("Ask a question about the PDF")

    if question:
        with st.spinner("Thinking..."):
            question_vector = question_embedding(question)
            top_chunks = faiss_search(
                question_vector,
                st.session_state.index,
                st.session_state.metadata
            )

            answer = generate_answer(question, top_chunks)

            pages = sorted(set(chunk["page"] for chunk in top_chunks))

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": pages
            })

    st.divider()
    st.subheader("💬 Conversation")

    for chat in st.session_state.chat_history:
        st.markdown("**🧑 You:**")
        st.write(chat["question"])

        st.markdown("**🤖 Assistant:**")
        st.write(chat["answer"])

        st.markdown("📌 *Sources:*")
        for p in chat["sources"]:
            st.write(f"- Page {p}")

        st.divider()

        #answer
        #st.markdown("Answer")
        #st.write(answer)

        #sources
        #pages = sorted(set(chunk["page"] for chunk in top_chunks))
        #st.markdown("Sources")
        #for p in pages:
         #   st.write(f"• Page {p}")


#package as desktop app