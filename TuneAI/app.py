import os
import sqlite3
import secrets
import torch
from torch.nn import functional as F
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoTokenizer

from model_arch import TuneAiModel, TuneAiConfig

app = FastAPI(title="TuneAi 35M API")

# --- DATABASE & SECURITY SETUP ---
DB_FILE = "api_keys.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS keys (api_key TEXT PRIMARY KEY)")
    # Inject the hardcoded development key to survive Hugging Face container reboots
    c.execute("INSERT OR IGNORE INTO keys (api_key) VALUES ('turnit_dev_key_2026')")
    conn.commit()
    conn.close()

init_db()

def verify_admin(x_admin_secret: str = Header(...)):
    expected_secret = os.environ.get("ADMIN_SECRET", "super_secret_admin_2026")
    if x_admin_secret != expected_secret:
        raise HTTPException(status_code=403, detail="Invalid Admin Secret")
    return True

def verify_api_key(x_api_key: str = Header(...)):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT 1 FROM keys WHERE api_key = ?", (x_api_key,))
    result = c.fetchone()
    conn.close()
    if not result:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# --- MODEL INFERENCE SETUP ---
tokenizer = AutoTokenizer.from_pretrained("gpt2")

try:
    with open('dataset/meta.txt', 'r') as f:
        vocab_size = int(f.read().strip())
except FileNotFoundError:
    vocab_size = tokenizer.vocab_size

config = TuneAiConfig(vocab_size=vocab_size)
model = TuneAiModel(config)

# Load the new 106MB checkpoint safely onto the CPU
model.load_state_dict(torch.load('checkpoints/tuneai_v1.pth', map_location=torch.device('cpu')))
model.eval()

# --- API ENDPOINTS ---
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.8

@app.post("/admin/generate-key")
def generate_key(admin_verified: bool = Depends(verify_admin)):
    new_key = f"tk_{secrets.token_hex(16)}"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO keys (api_key) VALUES (?)", (new_key,))
    conn.commit()
    conn.close()
    return {"message": "API Key generated successfully", "api_key": new_key}

@app.post("/v1/generate")
def generate_text(req: GenerateRequest, key_verified: bool = Depends(verify_api_key)):
    # 1. Encode prompt using the new Sub-Word Tokenizer (Replaces 'stoi')
    idx = torch.tensor(tokenizer.encode(req.prompt), dtype=torch.long).unsqueeze(0)
    
    # 2. Autoregressive text generation loop
    with torch.no_grad():
        for _ in range(req.max_tokens):
            # Crop context to the block_size defined in your architecture (256)
            idx_cond = idx[:, -config.block_size:]
            
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / req.temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop early if the model generates the End-Of-Text token
            if idx_next.item() == tokenizer.eos_token_id:
                break

    # 3. Decode sub-words back into a human-readable string (Replaces 'itos')
    out_text = tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)
    
    return {
        "model": "TuneAi-35M",
        "prompt": req.prompt,
        "generated_text": out_text
    }
