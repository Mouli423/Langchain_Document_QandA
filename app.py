import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")#
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Document Q and A "
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")

with st.expander("ℹ️ About this App"):
    st.markdown("""
    **Document Q&A System (RAG-based)**
    
    This application combines *Retrieval-Augmented Generation (RAG)* with state-of-the-art **Large Language Models (LLMs)** 
    to provide accurate, context-aware answers from uploaded research papers.
                
    The current knowledge base is built using the following research papers:
    - **attention.pdf** — Based on the landmark paper *"Attention Is All You Need"* (Vaswani et al., 2017), introducing the Transformer architecture.
    - **llm.pdf** — Covering modern advancements in *Large Language Models* and their applications in reasoning and generation.


    Under the hood, this app leverages:
    - **LLM**: LLaMA 3.1 (via Groq API)
    - **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
    - **Retriever**: FAISS vector database
    - **Framework**: LangChain for chaining prompts, retrieval, and responses
    """)


llm=ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-8b-instant")

if "vectors" not in st.session_state:
    st.session_state.vectors = None

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user queries based on the provided context only."
    " Please provide accurate responses."),
    ("human", """<context>
{context}
<context>
Question: {input}""")
])

def create_vector_embeddings():
    if "Embeddings" not in st.session_state:
        try:
            st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            loader=PyPDFDirectoryLoader("research_papers")
            docs=loader.load()

            if not docs:
                st.warning('No documents found in the provided directory')

            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=250)
            final_documents=text_splitter.split_documents(docs)
            st.session_state.vectors=FAISS.from_documents(final_documents,st.session_state.embeddings)

            st.success("Documents embedded and stored into vector DB sucessfully.")

        except Exception as e:
            st.error(f'Failed to create embeddings :{e}')
            st.session_state.vectors=None

st.title("RAG document Q and A with groq model, huggingface embeddings and FAISS Vector Database")

user_prompt=st.text_input("Enter your questions from the research papers")


try:
    if user_prompt:

        if st.session_state.vectors is None:
            with st.spinner("Creating document embeddings before answering..."):
                create_vector_embeddings()  
        if st.session_state.vectors:
            retriever = st.session_state.vectors.as_retriever()
            document_chain=create_stuff_documents_chain(llm,prompt)
            retrieval_chain=create_retrieval_chain(retriever,document_chain)
            start=time.process_time()
            response=retrieval_chain.invoke({"input":user_prompt})
            print(f"Response Time :{time.process_time()-start}")
            st.write(response["answer"])


except Exception as e:
    st.error(f'Error occured {e}')