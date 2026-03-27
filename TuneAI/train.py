import torch
import os
from model_arch import TuneAIModel
from data_prepare import prepare_data

# Configurations
batch_size = 32
max_iters = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Prepare Data
text, stoi, itos = prepare_data()
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]

# 2. Initialize Model
model = TuneAIModel(len(stoi)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 3. Training Loop
if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
print(f"TuneAI training starting on {device}...")

for iter in range(max_iters):
    # Sample a random batch of data
    ix = torch.randint(len(train_data) - 256, (batch_size,))
    x = torch.stack([train_data[i:i+256] for i in ix]).to(device)
    y = torch.stack([train_data[i+1:i+257] for i in ix]).to(device)

    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f"Step {iter}: Current Loss {loss.item():.4f}")
        torch.save(model.state_dict(), 'checkpoints/tuneai_v1.pth')

print("Success! Model saved to checkpoints/tuneai_v1.pth")
