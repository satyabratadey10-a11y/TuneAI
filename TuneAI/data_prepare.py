import os
import torch

def generate_build_tools_dataset(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    dataset_content = """[GRADLE_BUILD_SYSTEM]
Gradle is a build automation tool used by Android Studio.
It compiles resources and source code into APKs or AABs.
Tasks include assembling, compiling, and packaging.

[AAPT_TOOL]
aapt (Android Asset Packaging Tool) compiles AndroidManifest.xml and XML resources.
It generates the R.java class so you can reference resources in Java/Kotlin code.

[AAPT2_TOOL]
AAPT2 parses, indexes, and compiles Android resources into a binary format.
It supports incremental compilation by dividing the process into two steps: Compile and Link.
This dramatically improves build speed compared to the older aapt.

[APT_ANNOTATION_PROCESSING]
APT (Annotation Processing Tool) generates code at compile time.
For Kotlin, kapt and KSP (Kotlin Symbol Processing) handle these annotations.
Commonly used with libraries like Room, Dagger, and Hilt.
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(dataset_content.strip())
    print(f"Auto-generated Android Build Tools dataset at {file_path}")

def prepare_data():
    dataset_dir = 'dataset'
    file_path = os.path.join(dataset_dir, 'llms-full.txt')
    
    # Self-healing logic: Create dataset if it doesn't exist
    if not os.path.exists(dataset_dir) or not any(f.endswith('.txt') for f in os.listdir(dataset_dir)):
        print("No text files found in dataset/. Generating Build Tools Set...")
        generate_build_tools_dataset(file_path)

    text = ""
    for file in os.listdir(dataset_dir):
        if file.endswith('.txt'):
            with open(os.path.join(dataset_dir, file), 'r', encoding='utf-8') as f:
                text += f.read() + "\n"

    if not text:
        raise ValueError("Dataset is empty even after generation attempt.")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    print(f"Data prepared. Vocabulary size: {vocab_size}")
    return text, stoi, itos

if __name__ == "__main__":
    prepare_data()
