import hashlib
import io
import logging
import traceback
from pathlib import Path
from typing import List, Optional

import chromadb
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import requests
import urllib3
from chromadb import PersistentClient

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---- Setup FastAPI ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Chroma Setup ----
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("documents")
logger.info("Chroma vector store initialized.")

# ---- Models ----
class InferenceRequest(BaseModel):
    message: str
    history: Optional[List[List[str]]] = []

# ---- RAG Functions ----
def add_document(content: str, metadata: str = "") -> bool:
    try:
        embedding = embedding_model.encode(content).tolist()
        doc_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[{"filename": metadata}]
        )
        logger.info(f"Document '{metadata}' added with ID {doc_id}")
        return True
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        logger.debug(traceback.format_exc())
        return False

def get_similar_docs(query: str, k: int = 3) -> List[str]:
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents"]
        )
        docs = results["documents"][0]
        logger.info(f"Retrieved {len(docs)} similar docs for query: {query}")
        return docs
    except Exception as e:
        logger.error(f"Error retrieving similar docs: {e}")
        logger.debug(traceback.format_exc())
        return []



def call_rgt_llm(prompt: str) -> str:
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000,
    }
    headers = {
        "Authorization": f"Bearer {RGT_API_KEY}",
        "Content-Type": "application/json"
    }
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        response = requests.post(RGT_API_URL, json=payload, headers=headers, verify=False, timeout=30)
        result = response.json()
        message = result.get("choices", [{}])[0].get("message", {}).get("content", "No content returned.")
        logger.info("LLM responded successfully.")
        return message
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

# ---- Routes ----
@app.post("/inference/")
async def inference(request: InferenceRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty input")
    logger.info(f"Inference requested: {request.message}")

    docs = get_similar_docs(request.message, k=3)
    if docs:
        context = "\n---\n".join(docs)
        prompt = (
            f"You are an AI assistant. Use the context to answer the question.\n"
            f"Context:\n{context}\n"
            f"Question: {request.message}"
        )
    else:
        prompt = f"No context available. Answer this: {request.message}"

    response = call_rgt_llm(prompt)
    return {"response": response}

@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        logger.info(f"Received file: {file.filename}")

        if file.filename.endswith(".pdf"):
            pdf_reader = PdfReader(io.BytesIO(contents))
            text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif file.filename.endswith(".txt"):
            text = contents.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

        if not add_document(text, metadata=file.filename):
            raise HTTPException(status_code=500, detail="Failed to store document")

        return {"message": "File uploaded and indexed."}
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ---- Run ----
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI app...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
