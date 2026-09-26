import urllib.request
import json
import zipfile
import os
import ssl

def setup():
    print("Finding latest llama.cpp release with binaries...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request("https://api.github.com/repos/ggerganov/llama.cpp/releases")
    with urllib.request.urlopen(req, context=ctx) as response:
        releases = json.loads(response.read().decode())
        
    download_url = None
    for release in releases:
        for asset in release.get("assets", []):
            name = asset["name"].lower()
            if "cudart" in name:
                continue
            if name.startswith("llama-b") and "win-cuda" in name and name.endswith(".zip"):
                download_url = asset["browser_download_url"]
                break
            if name.startswith("llama-b") and "win-cu12" in name and name.endswith(".zip"):
                download_url = asset["browser_download_url"]
                break
        if download_url:
            break
            
    if not download_url:
        print("Could not find a valid llama.cpp windows zip.")
        return
        
    print(f"Downloading from {download_url}...")
    zip_path = "llama.zip"
    urllib.request.urlretrieve(download_url, zip_path)
    
    print("Extracting...")
    extract_path = "llama_cpp_bin"
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        
    os.remove(zip_path)
    
    # The zip usually contains a subfolder. Let's move files up or just find llama-server.exe
    exe_path = None
    for root, dirs, files in os.walk(extract_path):
        if "llama-server.exe" in files:
            exe_path = os.path.join(root, "llama-server.exe")
            break
            
    if exe_path:
        print(f"Standalone server ready at: {exe_path}")
        # Update ps1 script
        with open("start_local_llm.ps1", "w") as f:
            f.write(f'''$MODEL_PATH = "models\\qwen2-1_5b-instruct-q4_k_m.gguf"
$N_GPU_LAYERS = 999
$N_CTX = 8192
echo "Starting local OpenAI-compatible server on http://localhost:8000"
& "{exe_path}" -m $MODEL_PATH -ngl $N_GPU_LAYERS -c $N_CTX --host 0.0.0.0 --port 8000
''')
        print("Updated start_local_llm.ps1 successfully!")
    else:
        print("llama-server.exe not found in the extracted zip.")

if __name__ == "__main__":
    setup()
