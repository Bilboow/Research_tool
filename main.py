import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("AGENTS_API_KEY"),
    temperature=0.7,
    max_tokens=500,
)

st.title("LiteBot : Article Research Tool")
st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_button = st.sidebar.button("Process")
file_path = "./chroma_store"

main_placeholder = st.empty()

if process_url_button:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data from URLs... ✅")
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    main_placeholder.text("Splitting text... ✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and store in Chroma
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore_chroma = Chroma.from_documents(
        documents=docs,
        embedding=embeddings_model,
        persist_directory=file_path,
    )
    vectorstore_chroma.persist()

    main_placeholder.text("Embedding vector database created successfully ✅")
    time.sleep(2)

# Question input
query = main_placeholder.text_input("Ask a question about the articles:")

if query:
    if os.path.exists(file_path):
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore_chroma = Chroma(
            persist_directory=file_path,
            embedding_function=embeddings_model,
        )

        retriever = vectorstore_chroma.as_retriever()

        # Define the prompt
        prompt_template = """You are a research assistant.
Given the following context, answer the question based only on the context.
If the answer cannot be found, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        # Get response
        result = qa_chain.invoke({"query": query})

        st.header("Answer")
        st.write(result["result"])

        # Show sources
        st.subheader("Sources:")
        for doc in result["source_documents"]:
            st.write(doc.metadata.get("source", "Unknown"))
