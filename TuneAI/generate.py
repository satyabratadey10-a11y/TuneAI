import torch
import os
from model_arch import TuneAIModel, TuneAiConfig
from data_prepare import prepare_data

def generate_code():
    device = 'cpu'
    
    # 1. Load Data mapping
    print("Loading vocabulary...")
    _, stoi, itos = prepare_data()
    
    # 2. Initialize Model
    config = TuneAiConfig(vocab_size=len(stoi))
    model = TuneAIModel(config)
    
    model_path = 'checkpoints/tuneai_v1.pth'
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Make sure you are running this after training.")
        return

    print("Loading weights...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Generate starting with a specific prompt
    start_context = "[COMPOSE_CHAT_UI]"
    # Ensure characters exist in vocab
    idx = torch.tensor([[stoi.get(c, 0) for c in start_context if c in stoi]], dtype=torch.long, device=device)
    
    print(f"\n--- TuneAi Output ---")
    print(start_context, end='')
    with torch.no_grad():
        for _ in range(300): # Generate 300 characters
            logits, _ = model(idx)
            logits = logits[:, -1, :] 
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            print(itos[next_token.item()], end='', flush=True)
    print("\n--- End ---")

if __name__ == "__main__":
    generate_code()
