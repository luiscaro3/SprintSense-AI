from pydantic import BaseModel
import requests
from fastapi import FastAPI

app = FastAPI()

# Create a Pydantic model for the request body
class AddRequest(BaseModel):
    doc_id: str
    text: str


@app.post("/add")
def ingest_document(request: AddRequest):
    doc_id = request.doc_id
    text = request.text
    embedding = requests.post("http://localhost:8004/embed", json={"text": text}).json()["embedding"]
    requests.post("http://localhost:8000/add", json={"doc_id": doc_id, "embedding": embedding})



from pydantic import BaseModel
import requests
from fastapi import FastAPI
from typing import List, Optional
from datetime import datetime

app = FastAPI()

# Create a Pydantic model for the request body
class AddRequest(BaseModel):
    doc_id: str
    text: str  # Unstructured Jira story text

@app.post("/add")
def ingest_document(request: AddRequest):
    # Step 1: Extract structured attributes from unstructured text using the LLM
    extraction_response = requests.post(
        "http://localhost:8005/extract_attributes",  # Assuming you have an endpoint for attribute extraction
        json={"text": request.text}
    )
    extraction_response.raise_for_status()
    attributes = extraction_response.json()

    # Expected 'attributes' structure from LLM extraction:
    # {
    #     "title": "Fix login issue",
    #     "story_points": 5,
    #     "issue_type": "Bug",
    #     "priority": "High",
    #     "labels": ["login", "urgent"],
    #     "components": ["frontend", "backend"],
    #     "status": "In Progress",
    #     "created_date": "2024-02-19T10:00:00Z",
    #     "updated_date": "2024-02-20T14:00:00Z",
    #     "assignee": "john_doe"
    # }

    # Step 2: Generate embedding from the original text
    embedding_response = requests.post("http://localhost:8004/embed", json={"text": request.text})
    embedding_response.raise_for_status()
    embedding = embedding_response.json().get("embedding")

    if not embedding:
        return {"error": "Failed to generate embedding."}
    
    # Step 3: Prepare payload for vector DB ingestion
    payload = {
        "doc_id": request.doc_id,
        "embedding": embedding,
        "metadata": {
            "title": attributes.get("title") or "",
            "description": attributes.get("description") or "",
            "acceptance_criteria": ", ".join(attributes.get("acceptance_criteria", [])),
            "team": attributes.get("team") or "",
            "story_points": attributes.get("story_points"),
            "issue_type": attributes.get("issue_type") or "",
            "priority": attributes.get("priority") or "",
            "labels": ", ".join(attributes.get("labels", [])),
            "components": ", ".join(attributes.get("components", [])),
            "status": attributes.get("status") or "",
            "created_date": attributes.get("created_date") or "",
            "updated_date": attributes.get("updated_date") or "",
            "assignee": attributes.get("assignee") or ""
        }
    }

    # If description and acceptance criteria are the same, remove the description
    if payload["metadata"].get("description") == payload["metadata"].get("acceptance_criteria"):
        payload["metadata"].pop("description")


    # Step 4: Send data to the vector database
    vector_db_response = requests.post("http://localhost:8000/add", json=payload)
    vector_db_response.raise_for_status()



    return {"message": "Document ingested successfully with extracted attributes.", "data": vector_db_response.json()}
