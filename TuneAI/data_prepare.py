import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare():
    os.makedirs('dataset', exist_ok=True)
    print("1. Loading Python coding dataset...")
    # Downloading 10% of a 25k instruction-code dataset to stay under GitHub's 6-hour CPU limit
    dataset = load_dataset("flytech/python-codes-25k", split="train[:10%]")
    
    print("2. Initializing GPT-2 BPE Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Format the data into a conversational structure
    text_data = ""
    for item in dataset:
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        text_data += f"<|user|>\n{instruction}\n<|assistant|>\n{output}\n<|endoftext|>\n"
        
    print(f"3. Tokenizing raw code...")
    tokens = tokenizer.encode(text_data, add_special_tokens=False)
    
    # 90% Training, 10% Validation split
    n = len(tokens)
    train_data = tokens[:int(n*0.9)]
    val_data = tokens[int(n*0.9):]
    
    print("4. Compressing and saving binary blocks...")
    # GPT2 vocab size is 50257, which perfectly fits inside a uint16 (max 65535) saving massive RAM
    np.array(train_data, dtype=np.uint16).tofile('dataset/train.bin')
    np.array(val_data, dtype=np.uint16).tofile('dataset/val.bin')
    
    # Save a metadata file so train.py knows exactly how many outputs the neural net requires
    with open('dataset/meta.txt', 'w') as f:
        f.write(str(tokenizer.vocab_size))
        
    print(f"Data preparation complete. Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

if __name__ == '__main__':
    prepare()
