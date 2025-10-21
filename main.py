import os
import time
import pickle
import langchain
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

from dotenv import load_dotenv
load_dotenv()

# ---------------------- Page Config -------------------------
st.set_page_config(page_title="Research Tool", page_icon="🔍", layout="wide")
st.markdown("<h1 style='text-align: center;'>🔍 Research Tool</h1>", unsafe_allow_html=True)

# st.title('🔍 Research Tool')
st.markdown(
    "<hr style='border: 1.8px solid #3C3C3C; margin-top: -10px; margin-bottom: 20px;'>",
    unsafe_allow_html=True
)

st.sidebar.title('🌐 Source Links')

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"🔗 URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button('⚡ Process URL')

main_placeholder = st.empty()

llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.9
)

if process_url_clicked:
    # Load data ⏳
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("⏳ Data Loading Started...")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," "],
        chunk_size=1000
    )
    main_placeholder.text("⏳ Text Splitter Started...")
    docs = text_splitter.split_documents(data)

    # Creating embeddings and save to FAISS index
    embeddings = HuggingFaceEmbeddings()
    main_placeholder.text("⏳ Embedding Vector Started Building...")
    vector_index = FAISS.from_documents(docs, embeddings)

    file_path = "./stored_vectors/vector_index.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)
    main_placeholder.text("✅ Url Processing Done.")

file_path = "./stored_vectors/vector_index.pkl"

st.markdown(
    """
    <style>
    .compact-label {
        font-size: 25px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: -30px;
    }
    </style>
    <div class='compact-label'>📝 Question:</div>
    """,
    unsafe_allow_html=True
)
query = st.text_input("", placeholder="Ask your question...")

# Button to trigger the LLM
ask_button = st.button("Ask Question")

if ask_button and query:  # Only run when button clicked and query is not empty
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_index = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vector_index.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)

            # Display Answer
            st.subheader("🧠 Answer:")
            st.write(result["answer"])

            # Display Sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("🌐 Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)

# ---------------------- Footer -------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 130px;
        width: 100%;
        border-top: 1px solid #444;
        background-color: #1f1f1f;  /* dark background */
        color: #B0B0B0;
        text-align: center;
        font-size: 14px;
        line-height: 1.6;
        padding: 10px 0;
        z-index: 9999;
    }
    .footer a {
        color: #B0B0B0;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="footer">
        🔍 Research Tool &nbsp;&nbsp;|&nbsp;&nbsp;
        Built by <b>Niloy Sannyal</b> <br>
        Email: <a href="mailto:niloysannyal@gmail.com">niloysannyal@gmail.com</a> &nbsp;&nbsp;|&nbsp;&nbsp;
        GitHub: <a href="https://github.com/niloysannyal" target="_blank">github.com/niloysannyal</a> <br>
        &copy; 2025 All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
