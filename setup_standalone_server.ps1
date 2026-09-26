$ModelUrl = "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf"
$ModelPath = "models\qwen2-1_5b-instruct-q4_k_m.gguf"
$LlamaExtractedPath = "llama_cpp_bin"
$LlamaZipPath = "llama.zip"

New-Item -ItemType Directory -Force -Path "models"

# 1. Download the GGUF model directly
if (-Not (Test-Path $ModelPath)) {
    Write-Host "Downloading extremely efficient Qwen model..."
    Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelPath
    Write-Host "Model downloaded successfully."
}

# 2. Download standalone llama.cpp server for Windows (CUDA 12)
if (-Not (Test-Path $LlamaExtractedPath)) {
    Write-Host "Finding latest llama.cpp release..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    
    # Match the CUDA 12 version for Windows
    $asset = $release.assets | Where-Object { $_.name -match "win-cu12" -or $_.name -match "win-cuda" } | Select-Object -First 1
    
    if ($asset) {
        Write-Host "Downloading $($asset.name)..."
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $LlamaZipPath
        Expand-Archive -Path $LlamaZipPath -DestinationPath $LlamaExtractedPath -Force
        Remove-Item $LlamaZipPath
        Write-Host "Standalone server extracted."
    } else {
        Write-Host "Error: Could not find the correct llama.cpp windows zip!"
    }
}

Write-Host "Setup complete! You can now run .\start_local_llm.ps1"
