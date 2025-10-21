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

st.title('🔍 Research Tool')

st.sidebar.title('Source Links')

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button('Process URL')

main_placeholder = st.empty()

llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.9
)

if process_url_clicked:
    # Load data ⏳
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("✅ Data Loading Started...")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," "],
        chunk_size=1000
    )
    main_placeholder.text("✅ Text Splitter Started...")
    docs = text_splitter.split_documents(data)

    # Creating embeddings and save to FAISS index
    embeddings = HuggingFaceEmbeddings()
    main_placeholder.text("✅ Embedding Vector Started Building...")
    vector_index = FAISS.from_documents(docs, embeddings)

    file_path = "./stored_vectors/vector_index.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)
    main_placeholder.text("✅ Url Processing Done.")

file_path = "./stored_vectors/vector_index.pkl"
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_index = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result = {"answer": "", "sources": []}
            st.subheader("Answer: ")
            st.write(result["answer"])

            # Display Sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
