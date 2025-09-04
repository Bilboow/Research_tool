## 📖 Project: LiteBot – Article Research Tool

LiteBot is a Streamlit-based research assistant that allows users to input article URLs, automatically fetch content, split it into chunks, create embeddings with HuggingFace models, and store them in ChromaDB.
Users can then ask questions about the content, and LiteBot will generate answers with relevant sources using LangChain + ChatGroq.


## ⚡ Features

Fetches and processes articles from given URLs.

Splits long texts into meaningful chunks for embeddings.

Stores embeddings persistently with Chroma Vector Store.

Uses HuggingFace Sentence Transformers for embeddings.

Provides accurate Q&A with sources using LangChain + Groq LLM.

Built with Streamlit for an interactive UI.


## ⚙️ Installation & Run

1.Clone the repo (or copy the code):
  git clone <your-repo-link>
  cd Research_tool

2.Create and activate virtual environment (recommended):
  python3 -m venv venv
  source venv/bin/activate  # Mac/Linux
  venv\Scripts\activate     # Windows

3.Install dependencies:
  pip install -r requirements.txt  

4.Set up API key:
  AGENTS_API_KEY=your_groq_api_key_here

5.Run the Streamlit app:
  streamlit run main.py


## 🖼️ Screenshot:

<img width="1364" height="837" alt="Screenshot 2025-09-04 at 12 09 44 AM" src="https://github.com/user-attachments/assets/4001774f-94b2-4b7f-8464-0e8b57ead31a" />
<img width="1364" height="837" alt="Screenshot 2025-09-04 at 12 14 52 AM" src="https://github.com/user-attachments/assets/492f58b5-dba7-41ec-94ec-891efb8a1dc1" />



