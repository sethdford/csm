import os
import torch
from transformers import AutoModelForCausalLM

def download_weights():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Download Llama-1B model
    model = AutoModelForCausalLM.from_pretrained("sesame/csm-1b", torch_dtype=torch.float32)
    
    # Save model weights
    model.save_pretrained("models/llama-1B.pth")

if __name__ == "__main__":
    download_weights() 