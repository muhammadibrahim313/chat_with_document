import os
import base64
import gc
import tempfile
import uuid

import streamlit as st
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

# Setting up custom fonts and colors
st.markdown(
    """
    <style>
    .title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.5rem;
        color: #4CAF50;
    }
    .subtitle {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.2rem;
        color: #333333;
    }
    .footer {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1rem;
        color: #757575;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

# Function to reset chat
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

# Function to display PDF file
def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%">
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.header("🔑 Set Your Cohere API Key")
    st.markdown("[Get one at Cohere](https://dashboard.cohere.com/api-keys)")
    API_KEY = st.text_input("Enter your API Key", type="password")

    uploaded_file = st.file_uploader("📁 Choose your `.pdf` file", type="pdf")

    # Process the uploaded PDF file
    if uploaded_file and API_KEY:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{st.session_state.id}-{uploaded_file.name}"

                st.write("Please wait! Indexing your document...")

                # Creating an index over loaded data
                if file_key not in st.session_state.get('file_cache', {}):
                    loader = SimpleDirectoryReader(
                        input_dir=temp_dir,
                        required_exts=[".pdf"],
                        recursive=True
                    )
                    docs = loader.load_data()

                    # Setting up llm & embedding model
                    llm = Cohere(api_key=API_KEY, model="command")
                    embed_model = CohereEmbedding(
                        cohere_api_key=API_KEY,
                        model_name="embed-english-v3.0",
                        input_type="search_query",
                    )

                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    # Create the query engine
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    # Setting up reranker
                    reranker = CohereRerank(
                        top_n=2, model="rerank-english-v2.0", api_key=API_KEY
                    )
                    query_engine.node_postprocessors.append(reranker)

                    # Customizing prompt template
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above, I want you to think step by step to answer the query in a crisp manner. If you don't know the answer, say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    st.session_state.file_cache[file_key] = query_engine

                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and display the PDF uploaded
                st.success("✅ Ready to Chat!")
                display_pdf(uploaded_file)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Chat header and clear button
col1, col2 = st.columns([6, 1])
with col1:
    st.title("🔍 Searchable Document Chatbot")
    st.markdown('<p class="subtitle">Welcome to our interactive document search chatbot! Feel free to upload your PDF document and start asking questions.</p>', unsafe_allow_html=True)
    st.markdown('<p class="footer">Acknowledgment: This project was created with the assistance of <a href="https://lightning.ai/lightning-ai/studios/rag-using-cohere-command-r?view=public&section=featured%3Futm_source%3Dakshay&tab=overview">Cohere Command</a></p>', unsafe_allow_html=True)
    st.markdown('<p class="footer">Made with ❤️ by Muhammad Ibrahim Qasmi</p>', unsafe_allow_html=True)
    st.markdown("Connect with me: [LinkedIn](https://www.linkedin.com/in/muhammad-ibrahim-qasmi-9876a1297/) 🌐 | [GitHub](https://github.com/muhammadibrahim313) 📜 | [Kaggle](https://www.kaggle.com/muhammadibrahimqasmi) 📊")

with col2:
    st.button("🔄 Clear Chat", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
