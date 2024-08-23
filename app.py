import os
import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

# allows nested access to the event loop
nest_asyncio.apply()

# put your API key here, find one at: https://dashboard.cohere.com/api-keys
API_KEY = 'ziEpsRreaJzBi5HUDap7gMecJWXX69O26Hf71Kxo'

# setup llm & embedding model
llm = Cohere(api_key=API_KEY, model="command-r-plus")

embed_model = CohereEmbedding(
    cohere_api_key=API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

# Function to process uploaded PDFs
def process_pdfs(uploaded_files):
    temp_dir = 'temp_pdf_directory'
    os.makedirs(temp_dir, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.read())

    loader = SimpleDirectoryReader(
                input_dir=temp_dir,
                required_exts=[".pdf"],
                recursive=True
            )
    docs = loader.load_data()

    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_documents(docs, show_progress=True)

    cohere_rerank = CohereRerank(api_key=API_KEY)
    Settings.llm = llm
    query_engine = index.as_query_engine(node_postprocessors=[cohere_rerank])

    return query_engine

# Streamlit UI
st.title("PDF Query System")
st.write("Upload PDF files and ask questions to extract information from them.")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
question = st.text_input("Enter your question here...")

if uploaded_files and question:
    query_engine = process_pdfs(uploaded_files)
    response = query_engine.query(question)
    st.write("Response:")
    st.write(response)
else:
    st.write("Please upload PDF files and enter a question to get started.")
