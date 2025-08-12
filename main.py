from dotenv import load_dotenv
load_dotenv()

import hashlib
import aiohttp
import tempfile
from pathlib import Path
import asyncio
from time import time

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

import re
from bs4 import BeautifulSoup
# FastAPI Setup

from langchain_community.document_loaders import SeleniumURLLoader

def webdata(url):
    loader = SeleniumURLLoader(urls=[url])
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)
app = FastAPI()
security = HTTPBearer()
API_TOKEN = "d6ff8907483707b9cff5da7038fc37325b38fca575c6297372743203a77aa1cd"

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# Load Model Once
llm_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1.0,
)

# llm_model = ChatGroq(
#     model="openai/gpt-oss-120b",  # "qwen/qwen3-32b",# "whisper-large-v3-turbo", #"gemma2-9b-it", #"llama-3.3-70b-versatile", #llama-3.1-8b-instant",  # or llama3-8b-8192 /
#     temperature=1.0 )

# Prompt Setup
system_prompt = """
You are an information retrieval assistant. Answer the user's question based only on the given context from the document. Provide a brief explanation related to the context instead of a single-word answer. Do not include any justification. Just return the answer as plain text.
If the answer is not found in the context, give answer according to the context.   
Context: {context}
"""
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

# In-Memory Cache for Vectorstores
VECTOR_CACHE: Dict[str, FAISS] = {}

# Utility: Hash PDF URL
def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

# Step 1: Load and Split PDF
async def load_documents_async(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Failed to download PDF")
            content = await response.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ". ", "? ", "! ", " "])
    return splitter.split_documents(pages)


async def fetch_webpage_text(url):
    """Fetch and clean webpage text."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                return ""
            html = await response.text()

    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.stripped_strings)

  # ADD THIS AT THE TOP

async def extract_urls_from_docs(docs):
    url_info = []
    url_pattern = re.compile(r'(https?://[^\s]+)')
    for doc in docs:
        found_urls = url_pattern.findall(doc.page_content)
        for url in found_urls:
            url_info.append({
                "url": url,
                "page_number": doc.metadata.get("page", "unknown"),
                "pdf_source": doc.metadata.get("source", "PDF Document")
            })
    return url_info

# Load PDF + linked website content

async def load_documents_with_links(url: str):
    docs = await load_documents_async(url)
    urls = await extract_urls_from_docs(docs)

    for info in urls:
        web_text = webdata(info["url"])
        if web_text.strip():
            docs.append(
                Document(
                    page_content=web_text,
                    metadata={
                        "source": info["url"],
                        "found_in_pdf_page": info["page_number"],
                        "pdf_source": info["pdf_source"]
                    }
                )
            )
    return docs

# Step 2: Create or Reuse Vectorstore
def get_or_create_vectorstore(docs, url: str) -> FAISS:
    key = hash_url(url)
    if key in VECTOR_CACHE:
        return VECTOR_CACHE[key]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/static-retrieval-mrl-en-v1")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # latest & fast
    # embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    VECTOR_CACHE[key] = vectorstore
    return vectorstore

# Step 3: Create Retrieval Chain
def build_chain(vectorstore: FAISS):
    document_chain = create_stuff_documents_chain(llm=llm_model, prompt=prompt_template)
    retriever = vectorstore.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

# Step 4: Get answer from chain
def get_answer(chain, question: str) -> str:
    return chain.invoke({"input": question})["answer"]

# Async Wrapper for Answering
async def get_answer_async(chain, question: str):
    return await asyncio.to_thread(get_answer, chain, question)

# API Endpoint
@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_submission(req: RunRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    start_time = time()
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    docs = await load_documents_with_links(req.documents)
    vector = get_or_create_vectorstore(docs, req.documents)
    chain = build_chain(vector)
    answers = await asyncio.gather(*[get_answer_async(chain, q) for q in req.questions])
    with open("output/docs.txt", "a", encoding="utf-8") as f:
        f.write(f"Document URL: {req.documents}\n\n")
        for doc in docs:
            f.write(doc.page_content + "\n\n")
    with open("output/ques.txt", "a", encoding="utf-8") as f:
        for question in req.questions:
            f.write(f"Q: {question}\n\n")
    with open("output/ans.txt", "a", encoding="utf-8") as f:
        for answer in answers:
            f.write(f"A: {answer}\n\n")
    # print(f"Documents : {req.documents}")
    # print(f"Questions: {req.questions}")
    # print(f"Answers: {answers}")
    end_time = time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    return RunResponse(answers=answers)
