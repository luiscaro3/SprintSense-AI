import json
import re
from pydantic import BaseModel
from fastapi import FastAPI
import requests

app = FastAPI()

# Create a Pydantic model for the request body
class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
def rag_pipeline(request: QueryRequest):
    query = request.query
    embedding = requests.post("http://localhost:8004/embed", json={"text": query}).json()["embedding"]
    docs = requests.post("http://localhost:8000/query", json={"embedding": embedding}).json()
    if not docs:
        return {"answer": "No documents found"}

    # Loop over docs['metadatas'][0] array and join the internal values into an array of strings
    context = ''
    for item in docs['metadatas'][0]:
        context += " ".join(f"{key}: {value}" for key, value in item.items()) + " "
    prompt = f"Given this Context: {context}\ How would you size this user story: {query}? Answer in a json format with three fields: 'story_points', 'confidence' and 'reasoning'."
    response = requests.post("http://localhost:8005/generate", json={"prompt": prompt}).json()["response"]

    # Regex pattern to extract JSON block
    pattern = r'\{\s*[^}]*\}'

    # Find all JSON-like objects in the response
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        json_str = matches[-1]  # Get the last JSON block
        try:
                extracted_json = json.loads(json_str)
                print("Extracted JSON:", extracted_json)
        except json.JSONDecodeError:
            print("Error: Found JSON block is not valid.")
    else:
        print("No JSON object found.")

    extracted_json['relative_story'] = docs['ids'][0][0]
    return extracted_json