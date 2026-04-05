import os
import numpy as np
import random
from transformers import AutoTokenizer

def generate_android_domain_knowledge():
    """Knowledge base containing every requested tool and language."""
    knowledge_base = {
        # Native & NDK Tools
        "lld": "The LLVM linker, used by the NDK to link C/C++ object files into shared libraries.",
        "ndk-build": "A build system based on GNU Make, used to compile native C/C++ code for Android via Android.mk.",
        "strip": "A tool that discards all symbols from object files to reduce the final APK/AAB size.",
        "ar": "An archive tool used to create, modify, and extract from archives (like static .a libraries).",
        "objdump": "Displays information about object files, useful for debugging compiled C++ binaries.",
        "gdb": "The GNU Debugger, used to debug native C/C++ code running on an Android device.",
        "llvm-rs-cc": "The RenderScript compiler that compiles .rs files into bytecode.",
        "bcc": "The RenderScript bytecode compiler.",
        
        # Core Android SDK & Build Tools
        "aapt": "Android Asset Packaging Tool. Compiles AndroidManifest.xml and XML resources.",
        "aapt2": "The newer, faster, and more stable asset packager that parses, compiles, and links resources.",
        "aar": "Android Archive format. Contains compiled code, resources, and an Android manifest.",
        "aidl": "Android Interface Definition Language compiler. Generates Java/Kotlin interfaces for IPC.",
        "aidl_cpp": "Generates C++ bindings for AIDL interfaces for native IPC communication.",
        "apksigner": "A tool that signs APKs and verifies that an APK's signature will be accepted on devices.",
        "apksigtool": "A tool for verifying APK signatures and v2/v3 signing schemes.",
        "zipalign": "Optimizes APK alignment to 4-byte boundaries to reduce RAM usage during app execution.",
        "bundletool": "A command-line tool to build Android App Bundles (AAB) and convert AABs into an APK set.",
        "d8": "The modern Android dex compiler that converts Java/Kotlin bytecode into Dalvik bytecode (.dex). Replaces dx.",
        "dx": "The older, deprecated Dex compiler. Superseded by d8.",
        "dexdump": "A tool to inspect the contents of .dex files.",
        "dexlist": "Lists all methods in a .dex file.",
        "dexmerge": "Merges multiple .dex files into a single or multiple larger .dex files.",
        
        # Emulation, Optimization & CLI
        "proguard": "An optimizer and obfuscator that shrinks, optimizes, and obfuscates Java/Kotlin bytecode.",
        "R8": "The default Android optimizer and obfuscator that integrates desugaring, shrinking, and dexing into one step.",
        "sdkmanager": "A command-line tool that allows you to view, install, update, and uninstall packages for the Android SDK.",
        "avdmanager": "A command-line tool that allows you to create and manage Android Virtual Devices (emulators).",
        "android": "A deprecated command-line tool previously used to manage the SDK and AVDs. Replaced by sdkmanager/avdmanager.",
        "lint": "A code quality and error checker tool that scans Android projects for potential bugs.",
        "resguard": "A resource shrinker that obfuscates resource names in the APK to reduce size.",
        "adb": "Android Debug Bridge. A versatile command-line tool that lets you communicate with an emulator or connected Android device.",
        "fastboot": "A diagnostic protocol and tool used to modify the flash filesystem via a USB connection.",
        "log tools": "Logcat and related tools used to dump a log of system messages.",
        "sqlite3": "A command-line program to access and manage SQLite databases created by Android apps.",
        "dmtracedump": "A tool that generates graphical call-stack diagrams from trace log files.",
        "etc1tool": "A command-line utility to compress PNG images to the ETC1 standard and decode ETC1 to PNG.",
        
        # Languages & JDK
        "JDK": "The Java Development Kit. Includes tools like javac (compiler), jar (archiver), and jlink (custom runtime builder).",
        "Kotlin": "A modern, statically typed programming language used as the primary language for Android development.",
        "C++": "A compiled systems programming language used via the Android NDK for high-performance mobile computing.",
        "JNI": "Java Native Interface. The bridge that allows Kotlin/Java code to call native C++ functions."
    }

    templates = [
        ("What does {tool} do?", "{tool} is {desc}"),
        ("Explain the purpose of {tool} in Android.", "In Android development, {tool} acts as {desc}"),
        ("How is {tool} used?", "Developers use {tool} because it is {desc}"),
        ("Define {tool}.", "{tool}: {desc}")
    ]

    print("1. Synthesizing domain-specific dataset...")
    raw_text = ""
    
    # Generate 15,000 conversational pairs to ensure the 35M parameter model memorizes the data
    for _ in range(15000):
        tool, desc = random.choice(list(knowledge_base.items()))
        q_template, a_template = random.choice(templates)
        
        question = q_template.format(tool=tool)
        answer = a_template.format(tool=tool, desc=desc)
        
        raw_text += f"<|user|>\n{question}\n<|assistant|>\n{answer}\n<|endoftext|>\n"
        
        # Occasionally inject code snippets to teach basic syntax formatting
        if tool == "Kotlin":
            raw_text += f"<|user|>\nWrite a basic Kotlin function.\n<|assistant|>\nfun main() {{\n    println(\"Hello Android\")\n}}\n<|endoftext|>\n"
        if tool == "C++":
            raw_text += f"<|user|>\nWrite a basic C++ JNI function.\n<|assistant|>\nextern \"C\" JNIEXPORT jstring JNICALL\nJava_com_turnit_ai_MainActivity_stringFromJNI(JNIEnv* env, jobject) {{\n    return env->NewStringUTF(\"Hello from C++\");\n}}\n<|endoftext|>\n"

    return raw_text

def prepare():
    os.makedirs('dataset', exist_ok=True)
    
    # Generate the synthetic data based on the user's explicit tool list
    text_data = generate_android_domain_knowledge()
    
    print("2. Initializing GPT-2 BPE Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("3. Tokenizing synthetic build-tools data...")
    tokens = tokenizer.encode(text_data, add_special_tokens=False)
    
    n = len(tokens)
    train_data = tokens[:int(n*0.9)]
    val_data = tokens[int(n*0.9):]
    
    print("4. Compressing and saving binary blocks...")
    np.array(train_data, dtype=np.uint16).tofile('dataset/train.bin')
    np.array(val_data, dtype=np.uint16).tofile('dataset/val.bin')
    
    with open('dataset/meta.txt', 'w') as f:
        f.write(str(tokenizer.vocab_size))
        
    print(f"Data preparation complete. Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

if __name__ == '__main__':
    prepare()
