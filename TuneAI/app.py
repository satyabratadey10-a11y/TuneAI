import os
import sqlite3
import uuid
import torch
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from model_arch import TuneAiModel, TuneAiConfig
from data_prepare import prepare_data

# Initialize FastAPI
app = FastAPI(title="TuneAi SLM API")

# Setup SQLite for API Keys
DB_FILE = "api_keys.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS keys (api_key TEXT PRIMARY KEY)''')
    # Insert a default dev key if empty
    c.execute("SELECT COUNT(*) FROM keys")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO keys (api_key) VALUES ('turnit_dev_key_2026')")
    conn.commit()
    conn.close()

init_db()

# Admin Secret to generate new keys (Set this in your hosting environment variables)
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "super_secret_admin_2026")

# Load Model Configuration & Weights
device = 'cpu'
try:
    print("Initializing TuneAi model...")
    _, stoi, itos = prepare_data()
    config = TuneAiConfig(vocab_size=len(stoi))
    model = TuneAIModel(config)
    
    model_path = 'checkpoints/tuneai_v1.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Success: TuneAi Model weights loaded.")
    else:
        print(f"CRITICAL WARNING: {model_path} not found. Ensure the checkpoint is uploaded.")
except Exception as e:
    print(f"Failed to initialize model: {e}")

# Request Structure
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0

# Security Dependencies
def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=403, detail="Missing X-API-Key header")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM keys WHERE api_key=?", (x_api_key,))
    result = c.fetchone()
    conn.close()
    if not result:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

def verify_admin(x_admin_secret: str = Header(None)):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Admin Secret")
    return x_admin_secret

# Endpoints
@app.post("/admin/generate-key")
def generate_api_key(admin: str = Depends(verify_admin)):
    """Generates a new API key. Requires the X-Admin-Secret header."""
    new_key = f"tk_{uuid.uuid4().hex}"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO keys (api_key) VALUES (?)", (new_key,))
    conn.commit()
    conn.close()
    return {"message": "API Key generated successfully", "api_key": new_key}

@app.post("/v1/generate")
def generate_text(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    """Core inference endpoint. Requires a valid X-API-Key header."""
    try:
        # Filter prompt to prevent crashes from unknown characters
        filtered_prompt = [c for c in request.prompt if c in stoi]
        if not filtered_prompt:
            raise HTTPException(status_code=400, detail="Prompt contains no valid vocabulary characters.")
            
        idx = torch.tensor([[stoi[c] for c in filtered_prompt]], dtype=torch.long, device=device)
        
        generated_text = ""
        with torch.no_grad():
            for _ in range(request.max_tokens):
                logits, _ = model(idx)
                logits = logits[:, -1, :] / request.temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
                
                char = itos[next_token.item()]
                generated_text += char
                
        return {
            "model": "TuneAi-1M",
            "prompt": "".join(filtered_prompt),
            "generated_text": generated_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "online", "model": "TuneAi 1M"}

if __name__ == "__main__":
    import uvicorn
    # Port 7860 is mandatory for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)
