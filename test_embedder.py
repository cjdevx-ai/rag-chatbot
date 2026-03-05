# test_embedder.py
import sys
import os

try:
    from rag.embedder import embed_query, embed_chunks
    
    # Test embedding a query
    print("Testing embed_query...")
    vector = embed_query("What is quantum computing?")
    print(f"  Vector length : {len(vector)}")
    print(f"  First 5 values: {vector[:5]}")

    # Test embedding a small batch of chunks
    print("\nTesting embed_chunks...")
    sample_texts = [
        "Quantum computers use qubits instead of classical bits.",
        "Climate change is driven by greenhouse gas emissions.",
    ]
    vectors = embed_chunks(sample_texts)
    print(f"  Number of vectors : {len(vectors)}")
    print(f"  Each vector length: {len(vectors[0])}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    if "WinError 1114" in str(e):
        print("\n--- TROUBLESHOOTING TIP ---")
        print("This error (WinError 1114) is common on Windows when PyTorch/Transformers")
        print("tries to initialize a DLL and fails due to Power Management settings.")
        print("Try these steps:")
        print("1. Set your Windows Power Plan to 'High Performance'.")
        print("2. If you have an NVIDIA/AMD GPU, ensure 'Switchable Dynamic Graphics' is")
        print("   set to 'Maximize Performance' for Python.exe.")
        print("3. Install the latest Visual C++ Redistributables.")
