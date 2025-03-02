# -----------------------------------
# Dockerfile: Vector DB Service (ChromaDB)
# -----------------------------------
# File: vector_db/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart
COPY . .
CMD ["python3", "vector_db_service.py"]