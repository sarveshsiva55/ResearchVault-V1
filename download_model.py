import os
from huggingface_hub import hf_hub_download

def download_qwen_gguf():
    # Model repo and filename
    # Qwen2-1.5B is incredibly efficient and fits easily in 4GB VRAM while providing great reasoning.
    repo_id = "Qwen/Qwen2-1.5B-Instruct-GGUF"
    filename = "qwen2-1_5b-instruct-q4_k_m.gguf"
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Downloading {filename} (approx. 1.1 GB). This is fully offline once downloaded...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=models_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"Model downloaded successfully to: {model_path}")
    print("It is now ready for offline, efficient local inference!")

if __name__ == "__main__":
    download_qwen_gguf()
