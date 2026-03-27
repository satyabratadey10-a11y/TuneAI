import os
import json

def prepare_data():
    # Merge all.txt files in the dataset folder
    text = ""
    if not os.path.exists('dataset'): os.makedirs('dataset')
    
    for file in os.listdir('dataset'):
        if file.endswith('.txt'):
            with open(f'dataset/{file}', 'r', encoding='utf-8') as f:
                text += f.read() + "\n"

    if not text:
        raise ValueError("No text found in dataset/ folder. Add.txt files first!")

    # Create vocabulary from unique characters
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    if not os.path.exists('tokenizer'): os.makedirs('tokenizer')
    with open('tokenizer/vocab.json', 'w') as f:
        json.dump({'stoi': stoi, 'itos': itos}, f)
        
    print(f"Data prepared. Vocabulary size: {len(chars)}")
    return text, stoi, itos

if __name__ == "__main__":
    prepare_data()
