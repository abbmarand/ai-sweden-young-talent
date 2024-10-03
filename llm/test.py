import kagglehub

# Download latest version
path = kagglehub.model_download("mistral-ai/mistral/pyTorch/7b-instruct-v0.1-hf")

print("Path to model files:", path)