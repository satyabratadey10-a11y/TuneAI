import torch
import os
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from model_arch import TuneAIModel, TuneAiConfig
from data_prepare import prepare_data

# Initialize FastAPI
app = FastAPI(title="TuneAi SLM API")

# Security: Load your API Key from environment variables
# On your hosting platform, set a secret named: TUNEAI_API_KEY
API_KEY_SECRET = os.getenv("TUNEAI_API_KEY", "default_dev_key")

# 1. Load Data mapping and Model during startup
_, stoi, itos = prepare_data()
config = TuneAiConfig(vocab_size=len(stoi))
device = 'cpu' # Free hosting tiers usually don't provide GPU; CPU is fast enough for 1M params

model = TuneAIModel(config)
# Ensure you upload 'checkpoints/tuneai_v1.pth' to the server
if os.path.exists('checkpoints/tuneai_v1.pth'):
    model.load_state_dict(torch.load('checkpoints/tuneai_v1.pth', map_location=device))
model.to(device)
model.eval()

# Request Model
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0

# API Key Validation Dependency
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return x_api_key

@app.post("/v1/generate")
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    """
    Standard generation endpoint for TurnIt project.
    """
    try:
        # Encode the prompt
        idx = torch.tensor([[stoi.get(c, 0) for c in request.prompt]], dtype=torch.long, device=device)
        
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
                
                # Optional: stop if model generates an 'end' token if you have one
        
        return {
            "model": "TuneAi-1M-SLM",
            "prompt": request.prompt,
            "choices": [{"text": generated_text}]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "online", "model": "TuneAi 1M active"}
