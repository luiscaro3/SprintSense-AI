# -----------------------------------
# Dockerfile: LLM Service (Mistral)
# -----------------------------------
# File: llm_service/Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip git
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub
RUN huggingface-cli login --token $TOKEN --add-to-git-credential
COPY . .
CMD ["python3", "llm_service.py"]
