Hybrid PDF & Web Knowledge Extraction System
📌 Overview
This project is an AI-powered multi-source question answering engine designed to extract and retrieve information from PDF documents and linked web pages.
It enhances traditional PDF-based QA by detecting embedded URLs inside documents, fetching their web content, and combining both sources for more complete, accurate answers.

🚀 Features
PDF Document Parsing – Extracts text from PDF files using PyPDFLoader.
Embedded URL Detection – Scans document content for hyperlinks.
Linked Web Content Retrieval – Downloads and processes webpages referenced in the PDF.
Semantic Search – Uses FAISS vector store with HuggingFaceEmbeddings for efficient retrieval.
Context-Aware Answers – Combines PDF and web context to generate precise responses using LangChain.
FastAPI REST API – Exposes an endpoint for automated question answering.
Asynchronous Processing – Fetches and processes multiple sources in parallel for speed.
In-Memory Caching – Avoids reprocessing the same document URLs.

🛠 Tech Stack
Language: Python 3.10+
Frameworks: FastAPI, LangChain
Embedding Models: HuggingFace Sentence Transformers
Vector Store: FAISS
Web Scraping: aiohttp, BeautifulSoup4
Document Parsing: PyPDFLoader (LangChain)
LLM Backend: OpenAI GPT / Groq LLM
Deployment: Uvicorn

🔧 Installation

1. Clone this repository:

git clone https://github.com/yourusername/hybrid-pdf-web-qa.git
cd hybrid-pdf-web-qa

2.Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3.Install dependencies:

pip install -r requirements.txt

4.Set environment variables in .env:

OPENAI_API_KEY=your_api_key_here

5.Run the application:

uvicorn main:app --reload


📡 API Usage
Endpoint: POST /api/v1/hackrx/run
Request Body:
{
  "documents": "https://example.com/sample.pdf",
  "questions": [
    "What is the main topic of the document?",
    "Summarize the content."
  ]
}
Response:
{
  "answers": [
    "The document discusses...",
    "Summary: ..."
  ]
}
