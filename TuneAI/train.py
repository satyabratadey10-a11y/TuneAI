import os
import torch
import numpy as np
from model_arch import TuneAiModel, TuneAiConfig

def get_batch(split, block_size, batch_size):
    # Load memory-mapped binary array directly from disk (prevents RAM overload)
    data_path = f'dataset/{split}.bin'
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Read uint16 bytes and cast to standard int64 PyTorch tensors
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y

def main():
    os.makedirs('checkpoints', exist_ok=True)
    
    # Read vocab size dynamically to prevent mismatch crashes
    with open('dataset/meta.txt', 'r') as f:
        vocab_size = int(f.read().strip())
        
    config = TuneAiConfig(vocab_size=vocab_size)
    model = TuneAiModel(config)
    
    # Audit logic to confirm parameter constraints
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initializing TuneAi: ~{n_params / 1e6:.2f}M Parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    epochs = 2000  # Adjusted for GitHub Action runtime constraints
    batch_size = 8
    block_size = config.block_size
    
    model.train()
    print("Beginning Training Loop...")
    for iter in range(epochs):
        xb, yb = get_batch('train', block_size, batch_size)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Step {iter} | Training Loss: {loss.item():.4f}")
            
    print("Training complete. Packaging model weights...")
    torch.save(model.state_dict(), 'checkpoints/tuneai_v1.pth')
    print("Model saved to checkpoints/tuneai_v1.pth successfully.")

if __name__ == '__main__':
    main()
