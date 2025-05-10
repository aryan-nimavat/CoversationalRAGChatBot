import streamlit as st
from dotenv import load_dotenv
from torch import classes
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load environment variables
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Conversational Q&A With RAG PDF Uploads"

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Set up streamlit app
st.title("Conversational Q&A with RAG PDF Uploads")
st.write("Upload a PDF and ask questions about its content.")

# Input Groq API key
api_key = st.text_input("Enter your Groq API key", type="password")
if api_key:
    llm = ChatGroq(model_name = "llama-3.3-70b-versatile")
    session_id = st.text_input("Enter a session ID:", value="default_session")
    
    # Chat Interface
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True)
    
    # Process the uploaded PDF
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.read())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
            
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        splits = text_splitter.split_documents(documents)
        
        # Create a vector store from the chunks
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()
        
        contextualized_q_system_prompt = (
        "Given a chat history, a question, and context, "
        "generate an answer to the question based on the context. "
        "The answer should be concise and relevant to the question. "
        "If the question cannot be answered based on the context, "
        "respond with 'I don't know'. "
        )
        contextualized_q_system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualized_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_q_system_prompt)
        
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the question cannot be answered based on the context, respond with 'I don't know'."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=qa_chain,
        )
        
        def get_session_history(session_id:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key = "input",
            history_messages_key = "chat_history",
            output_messages_key = "answer",
        )
        
        question = st.text_input("Ask a question about the PDF:")
        if question:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": question},  
                config={"configurable": {"session_id": session_id}}
            )

            st.write("Answer:", response["answer"])