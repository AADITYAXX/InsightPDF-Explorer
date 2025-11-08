import streamlit as st # Streamlit for web app
from openai import AzureOpenAI #Azure OpenAI client
from dotenv import load_dotenv  #env variable
import os #for environment variables
import fitz  #vectorization
import tiktoken #tokenization
import chromadb #vector database
import pytesseract  #image OCR
from PIL import Image #image processing
import io #for byte stream handling
import base64 #for base64 encoding
import time  # Added for sleep function

# Defines phrases that indicate the answer wasn't found in the PDF
DENIAL_PHRASES = [
    "does not contain information",
    "not found in the pdf",
    "not mentioned in the document",
    "document does not provide information",
    "no relevant information in the pdf",
    "document does not provide details",
    "the answer is not available in the following pdf",
    "hello! how can i assist you today?"
]

# Loads environment variables from a .env file
def load_environment():
    load_dotenv()

# Creates an Azure OpenAI client using credentials from environment variables
def initialize_azure_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

# Configures the Streamlit page layout and title
def configure_streamlit():
    st.set_page_config(page_title="InsightPDF Explorer", layout="centered")
    st.markdown("<h1 style='text-align: center;'> InsightPDF Explorer </h1>", unsafe_allow_html=True)

# Defines the system prompt that guides the AI's behavior
def initialize_session_state():
    prompt = (
        "Case 1: Greetings - If the user inputs a greeting (e.g., 'Hi', 'Hello', 'Hey', 'Good morning', 'Good afternoon', 'Good night', 'Greetings', or similar casual salutations, case-insensitive): "
        "Respond only with: 'Hello! How can I assist you today?' and do not show any buttons ('View', 'Download', 'Yes', or 'No'). "
        "Case 2: PDF-Only Answers - If a PDF is uploaded, use only the PDF content to answer. If the answer is found in the PDF: "
        "Show View and Download buttons. If the answer is not found: Show a message: 'The answer is not available in the following PDF. Do you want to search outside the document?' and display Yes/No buttons. "
        "If Yes is clicked: Trigger handle_external_search(client) to search externally. Do not show View or Download buttons. Rerun the app. "
        "If No is clicked: Append 'Ask me your next query' to the conversation and chat history. Do not show View or Download buttons. Rerun the app. "
        "Case 3: No PDF Uploaded - For queries when no PDF is uploaded, answer using internal knowledge and do not show any buttons."
    )

    defaults = {  # Sets default values for session state variables that track:
        "conversation": [{"role": "system", "content": prompt}],   # - Conversation history
        "chat_history": [],   # - Chat display history
        "first_interaction": True,  
        "pdf_docs": [],      #  - Uploaded PDFs
        "pdf_index": None,      # - Vector index of PDF content
        "total_chunks": 0,      
        "last_user_query": None    
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_pdf_uploader():
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
        return uploaded_files

# - Extracts text from PDF using PyMuPDF (fitz)
# - Also performs OCR on any images found in the PDF using pytesseract
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            text += "\n" + pytesseract.image_to_string(image)
    return text

def chunk_and_embed(text, filename):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunk_size = 500
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]  # - Splits text into chunks of 500 tokens
    decoded_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    embeddings = client.embeddings.create(input=decoded_chunks, model="text-embedding-ada-002")  # - Creates embeddings for each chunk using Azure OpenAI
    vectors = [e.embedding for e in embeddings.data] 
    ids = [f"{filename}chunk{i}" for i in range(len(decoded_chunks))]
    return decoded_chunks, vectors, ids  # - Returns chunks, their embeddings, and generated IDs

def process_uploaded_pdfs(uploaded_files):
    chroma_client = chromadb.PersistentClient(path=".chroma")
    collection_name = "pdf_chunks"

    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception as e:
        st.warning(f"Warning: Could not delete collection: {e}")

    collection = chroma_client.get_or_create_collection(name=collection_name)
    st.session_state.pdf_docs.clear()
    all_chunks = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
        st.session_state.pdf_docs.append({"filename": filename, "data": io.BytesIO(file_bytes)})

        text = extract_text_from_pdf(file_bytes)
        chunks, vectors, ids = chunk_and_embed(text, filename)
        all_chunks.extend(chunks)

        collection.add(
            documents=chunks,
            embeddings=vectors,
            ids=ids,
            metadatas=[{"source": filename}] * len(chunks)
        )

    st.session_state.pdf_index = collection
    st.session_state.total_chunks = len(all_chunks)

# - When user chooses to search outside PDF, this function queries the AI without PDF context
# - Streams the response for better user experience
def handle_external_search(client):
    user_input = st.session_state.last_user_query
    if not user_input:
        msg = {"role": "assistant", "content": "No previous query found. Please ask a new question."}
        st.session_state.conversation.append(msg)
        st.session_state.chat_history.append(msg)
        return

    external_conversation = [
        {"role": "system", "content": "Answer the user's query using your internal knowledge, without relying on any PDF context."},
        {"role": "user", "content": user_input}
    ]

    full_response = ""
    with st.spinner("Searching outside the document..."):
        stream_response = client.chat.completions.create(
            model="gpt-4o",
            messages=external_conversation,
            stream=True,
            max_tokens=1000,
        )
        for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta:
                full_response += chunk.choices[0].delta.content or ""

    msg = {"role": "assistant", "content": full_response}
    st.session_state.conversation.append(msg)
    st.session_state.chat_history.append(msg)

def is_greeting(user_input):
    greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "greetings", "howdy", "yo", "hola", "salut", "ciao", "good night"
    ]
    return user_input.lower().strip() in greetings

# - Main function that handles user queries
# - When PDFs are uploaded, searches for relevant content using vector similarity
def handle_chat_input(client):
    user_input = st.chat_input("Ask something anything or ask me about the PDFs...")
    matched_source = None
    is_specific_match = True

    if user_input:
        st.session_state.first_interaction = False
        st.session_state.last_user_query = user_input
        st.session_state.conversation.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Handle greetings explicitly
        if is_greeting(user_input):
            assistant_msg = {
                "role": "assistant",
                "content": "Hello! How can I assist you today?",
                "is_specific_match": False
            }
            st.session_state.conversation.append(assistant_msg)
            st.session_state.chat_history.append(assistant_msg)
            return None, False, user_input

        context = ""
        if st.session_state.pdf_index:
            query_embedding = client.embeddings.create(
                input=[user_input],
                model="text-embedding-ada-002"
            ).data[0].embedding

            results = st.session_state.pdf_index.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            matched_docs = results["documents"][0]
            matched_metadatas = results["metadatas"][0]
            context = "\n\n".join(matched_docs)
            matched_source = matched_metadatas[0]['source'] if matched_metadatas else None

            query_lower = user_input.lower()
            context_lower = context.lower()
            if not any(word in context_lower for word in query_lower.split()):
                is_specific_match = False

        if context and is_specific_match:
            st.session_state.conversation.append({
                "role": "system",
                "content": f"Use the following context (from {matched_source}) to answer:\n\n{context}"
            })
        else:
            st.session_state.conversation.append({
                "role": "system",
                "content": "No relevant information was found in the PDF. Respond with: 'The answer is not available in the following PDF. Do you want to search outside the document?'"
            })

        full_response = ""
        with st.spinner("Thinking..."):
            stream_response = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.conversation,
                stream=True,
                max_tokens=1000,
            )
            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta:
                    full_response += chunk.choices[0].delta.content or ""

        response_lower = full_response.lower()
        query_words = user_input.lower().split()
        used_pdf_context = (
            context.strip() != "" and
            any(word in context.lower() for word in query_words) and
            not any(phrase in response_lower for phrase in DENIAL_PHRASES)
        )

        assistant_msg = {"role": "assistant", "content": full_response}
        if used_pdf_context and matched_source:
            assistant_msg["source"] = matched_source
            assistant_msg["is_specific_match"] = is_specific_match
        else:
            assistant_msg["is_specific_match"] = False
            assistant_msg["requires_external_search"] = "the answer is not available in the following pdf" in response_lower

        st.session_state.conversation.append(assistant_msg)
        st.session_state.chat_history.append(assistant_msg)

    return matched_source, is_specific_match, user_input

# - Displays the chat history with special handling for the latest message
# - Implements a typewriter effect for the latest assistant response
def display_chat(matched_source, is_specific_match, user_input):
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            is_latest = (i == len(st.session_state.chat_history) - 1 and msg["role"] == "assistant")
            if is_latest:
                # Stream only the latest assistant response with delay
                def char_generator():
                    for char in msg["content"]:
                        time.sleep(0.05)  # Delay of 0.05 seconds between characters
                        yield char
                st.write_stream(char_generator())
            else:
                # Display older messages instantly
                st.markdown(msg["content"], unsafe_allow_html=True)

            if (
                msg["role"] == "assistant"
                and msg.get("source")
                and msg.get("is_specific_match", False)
                and msg["source"] in [doc["filename"] for doc in st.session_state.pdf_docs]
                and not any(phrase in msg["content"].lower() for phrase in DENIAL_PHRASES)
            ):
                source_file = msg["source"]
                for file in st.session_state.pdf_docs:
                    if file["filename"] == source_file:
                        pdf_bytes = file["data"].getvalue()
                        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                        open_link = f"""
                            <a href="data:application/pdf;base64,{b64_pdf}"
                               target="_blank"
                               style="display: inline-block; padding: 8px 16px;
                                      background-color: #4CAF50; color: white;
                                      text-decoration: none; border-radius: 4px;
                                      font-weight: bold;">
                               📖 View PDF
                            </a>
                        """
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(open_link, unsafe_allow_html=True)
                        with col2:
                            st.download_button(
                                label="⬇ Download PDF",
                                data=pdf_bytes,
                                file_name=source_file,
                                mime="application/pdf",
                                key=f"download_{source_file}{msg['content'][:10]}{i}"
                            )
                        break
            elif (
                msg["role"] == "assistant"
                and msg.get("requires_external_search", False)
            ):
                yes_key = f"yes_pressed_{i}"
                no_key = f"no_pressed_{i}"

                if not st.session_state.get(yes_key) and not st.session_state.get(no_key):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Yes", key=f"yes_{i}", use_container_width=True):
                            handle_external_search(client)
                            msg["requires_external_search"] = False
                            st.session_state.chat_history[i]["requires_external_search"] = False
                            st.session_state[f"yes_pressed_{i}"] = True
                            st.rerun()
                    with col2:
                        if st.button("❌ No", key=f"no_{i}", use_container_width=True):
                            msg = {"role": "assistant", "content": "Ask me your next query"}
                            st.session_state[f"no_pressed_{i}"] = True
                            st.session_state.conversation.append(msg)
                            st.session_state.chat_history.append(msg)
                            st.rerun()

def main():
    load_environment()
    global client
    client = initialize_azure_client()
    configure_streamlit()
    initialize_session_state()
    uploaded_files = create_pdf_uploader()

    if uploaded_files:
        process_uploaded_pdfs(uploaded_files)

    matched_source, is_specific_match, user_input = handle_chat_input(client)
    display_chat(matched_source, is_specific_match, user_input)

# - Orchestrates the entire application flow
# - Initializes components, handles PDF uploads, processes chat, and displays results

if name == "main":
    main()


# ## Flow Overview
# 1. Initialization: Load environment variables, set up the UI, and initialize session state
# 2. PDF Processing: When PDFs are uploaded, extract text, chunk it, and create embeddings
# 3. Chat Handling: Process user queries, search PDF content when available, and generate responses
# 4. Response Display: Show conversation history with interactive elements (view/download PDF buttons, yes/no options)
