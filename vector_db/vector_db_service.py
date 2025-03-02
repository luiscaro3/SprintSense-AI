from typing import Dict, Any
from fastapi import FastAPI
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel
import logging

app = FastAPI()
client = chromadb.PersistentClient(path=".chromadb")
collection = client.get_or_create_collection("rag_collection2")

# Create a Pydantic model for the request body
class AddRequest(BaseModel):
    doc_id: str
    metadata: Dict[str, Any]
    embedding: list

class QueryRequest(BaseModel):
    embedding: list

@app.post("/add")
def add_doc(request: AddRequest):

    doc_id = request.doc_id
    embedding = request.embedding
    metadata = request.metadata
    logging.info(f"Added document with ID {doc_id} and embedding {embedding}")
    collection.add(ids=[doc_id], metadatas=metadata, embeddings=[embedding])
    return request

@app.post("/query")
def query(request: QueryRequest):
    embedding = request.embedding
    results = collection.query(query_embeddings=[embedding], n_results=3)
    return results

    # # # return results
    # all_records = collection.get()
    # return all_records

    # # Print existing documents, metadatas, and ids
    # for i, doc_id in enumerate(all_records["ids"]):
    #     print(f"ID: {doc_id}")
    #     print(f"Document: {all_records['documents'][i]}")
    #     print(f"Metadata: {all_records['metadatas'][i]}")