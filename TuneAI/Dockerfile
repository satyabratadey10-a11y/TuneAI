FROM python:3.10-slim

WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your AI API scripts and weights
COPY . .

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Force the container to boot Uvicorn directly, ignoring app.py execution quirks
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
