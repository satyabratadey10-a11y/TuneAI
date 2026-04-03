FROM python:3.10-slim

WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your AI API scripts and weights
COPY . .

# Expose the port Hugging Face requires
EXPOSE 7860

# Start the FastAPI server
CMD ["python", "app.py"]
