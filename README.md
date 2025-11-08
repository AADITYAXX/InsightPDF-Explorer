🧠 InsightPDF Explorer

Chat with Your PDFs — Powered by Azure OpenAI & Streamlit

InsightPDF Explorer is an interactive Streamlit-based web application that enables users to query, understand, and extract insights from one or more PDF documents.
It combines Azure OpenAI, ChromaDB, and OCR to deliver intelligent, context-aware answers directly from your documents — all through a natural chat interface.

🚀 Features

💬 Conversational Interface – Chat with your PDFs like talking to an assistant.

🧠 AI-Powered Contextual Answers – Uses Azure OpenAI GPT models to answer queries accurately.

📚 Multi-PDF Support – Upload multiple PDFs for cross-document querying.

🔍 Smart PDF Search – Extracts, chunks, and embeds text for semantic search using vector similarity.

🖼️ OCR Extraction – Reads text from images inside PDFs using Tesseract OCR.

⚡ Streaming Responses – Real-time answer streaming for a smooth chat experience.

📄 Interactive PDF Options – View or download relevant PDFs directly from the chat.

🌐 External Search Option – If the answer isn’t found in PDFs, it intelligently offers to search using AI’s knowledge.

🧩 Tech Stack
Component	Technology
Frontend UI	Streamlit

AI Model	Azure OpenAI (GPT-4 / GPT-4o)
Embeddings	text-embedding-ada-002
Vector Database	ChromaDB

OCR	pytesseract

PDF Processing	PyMuPDF (fitz)

Tokenization	tiktoken

Environment Variables	python-dotenv
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/AADITYAXX/JobCrate.git
cd JobCrate

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Up Environment Variables

Create a .env file in your project root and add your Azure OpenAI credentials:

AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

5️⃣ Run the App
streamlit run app.py


Replace app.py with your file name if different.

🧠 How It Works

Upload PDFs in the sidebar.

The app:

Extracts text and images from each PDF.

Runs OCR on images to get embedded text.

Splits text into tokenized chunks.

Generates embeddings via OpenAI’s text-embedding-ada-002.

Stores them in a ChromaDB vector index for similarity search.

Ask questions in the chat box.

If the answer exists in PDFs → it responds using PDF context.

If not → it asks whether to “search outside the document” using GPT’s internal knowledge.

View or download relevant PDFs right from the chat interface.

💻 Project Structure
📦 InsightPDF Explorer
├── app.py                     # Main Streamlit application
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables
├── .chroma/                    # Persistent ChromaDB vector storage
├── README.md                   # Project documentation
└── assets/                     # (Optional) logos, screenshots, etc.

🧰 Dependencies

Make sure these Python packages are installed (or via requirements.txt):

streamlit
openai
python-dotenv
PyMuPDF
tiktoken
chromadb
pytesseract
Pillow

🧾 Example Usage

Upload one or more PDFs via the sidebar.

Type a question like:

What are the main findings in the report?


The assistant responds using the uploaded document’s content.

If the information isn’t available, you’ll be prompted:

The answer is not available in the following PDF. Do you want to search outside the document?


You can click Yes (search externally) or No (continue chatting).


🧑‍💻 Author

Aaditya
GitHub: @AADITYAXX

⚖️ License

This project is licensed under the MIT License — free to use, modify, and distribute.
