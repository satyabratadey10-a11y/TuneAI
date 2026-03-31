import torch
from model_arch import TuneAIModel, TuneAiConfig
from data_prepare import prepare_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load Data mapping
_, stoi, itos = prepare_data()

# 2. Load Model
config = TuneAiConfig(vocab_size=len(stoi))
model = TuneAIModel(config)
model.load_state_dict(torch.load('checkpoints/tuneai_v1.pth', map_location=device))
model.to(device)
model.eval()

# 3. Generate
start_context = "implementation" # Or any start word from your dataset
idx = torch.tensor([[stoi.get(c, 0) for c in start_context]], dtype=torch.long, device=device)

print(f"--- TuneAi Generating from: '{start_context}' ---")
with torch.no_grad():
    # Generate 100 tokens
    for _ in range(100):
        logits, _ = model(idx)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        print(itos[next_token.item()], end='', flush=True)
print("\n--- End ---")
