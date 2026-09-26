$MODEL_PATH = "models\qwen2-1_5b-instruct-q4_k_m.gguf"
$N_GPU_LAYERS = 999
$N_CTX = 8192
echo "Starting local OpenAI-compatible server on http://localhost:8000"
& "llama_cpp_bin\llama-server.exe" -m $MODEL_PATH -ngl $N_GPU_LAYERS -c $N_CTX --host 0.0.0.0 --port 8000
