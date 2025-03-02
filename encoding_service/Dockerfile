# -----------------------------------
# Dockerfile: Encoding Service (BGE-M3)
# -----------------------------------
# File: encoding_service/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y git
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python3", "encoding_service.py"]