import streamlit as st
import os
import time
import tempfile

from dotenv import load_dotenv
load_dotenv()

# LLM
from langchain_groq import ChatGroq

# Embeddings (FREE)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector DB
from langchain_community.vectorstores import FAISS

# PDF Loaders
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

# Prompt + LCEL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")


# Session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# LLM (UPDATED MODEL )
llm = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the questions based only on the provided context.

<context>
{context}
</context>

Question: {input}
""")

# UI
st.title("RAG Q&A with PDF Upload (Groq + Free Embeddings)")

# Show chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"** You:** {msg}")
    else:
        st.markdown(f"** AI:** {msg}")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

#  Process PDFs (with OCR fallback)
def process_uploaded_files(files):

    documents = []

    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Try normal loader first
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Remove empty pages
        docs = [doc for doc in docs if doc.page_content.strip() != ""]

        # If no text → use OCR
        if len(docs) == 0:
            st.warning(f" Using OCR for {uploaded_file.name} (scanned PDF)")
            loader = UnstructuredPDFLoader(tmp_path)
            docs = loader.load()

        documents.extend(docs)

    return documents

# Button to process docs
if st.button("Process Documents"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    with st.spinner("Processing PDFs..."):

        docs = process_uploaded_files(uploaded_files)

        #  Safety check
        if not docs:
            st.error(" No readable content found in PDFs.")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_docs = text_splitter.split_documents(docs)

        if not final_docs:
            st.error(" No text chunks created.")
            st.stop()

        # Debug info
        st.write(f" Pages loaded: {len(docs)}")
        st.write(f" Chunks created: {len(final_docs)}")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vectors = FAISS.from_documents(
            final_docs,
            embeddings
        )

    st.success(" Documents processed successfully!")

# Query input
user_prompt = st.chat_input("Ask a question from your PDFs")


# Query execution
if user_prompt:

    if st.session_state.vectors is None:
        st.warning("Please upload and process documents first.")
        st.stop()

    #retriever = st.session_state.vectors.as_retriever()
    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 6}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    start = time.process_time()

    #response = rag_chain.invoke(user_prompt)
    response = rag_chain.invoke(user_prompt)
    
    # Save to chat history
    st.session_state.chat_history.append(("user", user_prompt))
    st.session_state.chat_history.append(("ai", response))

    st.write(response)
    st.write(f" Response time: {time.process_time()-start:.2f} sec")


    # Show retrieved chunks
    with st.expander("🔍 Retrieved Chunks"):
        docs = retriever.invoke(user_prompt)
        for doc in docs:
            st.write(doc.page_content)
            st.write("------")