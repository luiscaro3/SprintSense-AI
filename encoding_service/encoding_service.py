from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("BAAI/bge-m3")

# Create a Pydantic model for the request body
class QueryRequest(BaseModel):
    text: str


@app.post("/embed")
def embed(request: QueryRequest):
    text = request.text
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}