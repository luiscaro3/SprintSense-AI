# -----------------------------------
# Dockerfile: Intake Service
# -----------------------------------
# File: intake_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python3", "intake_service.py"]