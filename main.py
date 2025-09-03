import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key= os.getenv("AGENTS_API_KEY"),
    temperature= 0.9,
    max_tokens=500
)

st.title("LiteBot : Article research tool")
st.sidebar.title("Article URLs")

Urls=[]
for i in range(3):
   url = st.sidebar.text_input(f"input{i+1}")
   Urls.append(url)

process_url_button = st.sidebar.button("Process")
file_path = "./chroma_store"

main_placeholder = st.empty()


if process_url_button:
   #load data
   loader = UnstructuredURLLoader(urls = Urls)
   main_placeholder.text("Data Loading...Started...✅✅✅")
   data = loader.load()

   #data split
   text_splitter = RecursiveCharacterTextSplitter(
      separators=['\n','\n\n',' '],
      chunk_size = 1000
   )
   main_placeholder.text("Text Splitter...Started...✅✅✅")
   doc = text_splitter.split_documents(data)

   # creating embedding and save it in chroma
   embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
   vectorstore_chroma = Chroma.from_documents(
       doc,
       embeddings_model,
       persist_directory=file_path)
   #vectorstore_chroma.persist()

   main_placeholder.text("Embedding Vector Started Building...✅✅✅") 
   time.sleep(2)

query = main_placeholder.text_input("Question : ")   
if query:
   if os.path.exists(file_path):
      
      embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
      vectorstore_chroma = Chroma(persist_directory=file_path, 
                                  embedding_function=embeddings_model)

      combine_docs_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
      chain = RetrievalQAWithSourcesChain(
            retriever=vectorstore_chroma.as_retriever(),
            combine_documents_chain=combine_docs_chain,
            return_source_documents=True
        )
      result = chain({"question": query}, return_only_outputs= True)

      st.header("Answer")
      st.write(result['answer'])
      # Display sources, if available

      sources = result.get("sources", "")
      if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)            