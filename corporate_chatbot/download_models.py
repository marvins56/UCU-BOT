# download_models.py
import os
import requests
from tqdm import tqdm

MODELS = {
    "mistral": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    },
    "phi2": {
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi-2.Q4_K_M.gguf"
    },
    "bloomz": {
        "url": "https://huggingface.co/TheBloke/bloomz-560m-GGUF/resolve/main/bloomz-560m.Q4_K_M.gguf",
        "filename": "bloomz-560m.Q4_K_M.gguf"
    }
}

def download_file(url: str, filename: str):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    models_dir = './models'
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, model_info in MODELS.items():
        print(f"\nDownloading {model_name}...")
        filepath = os.path.join(models_dir, model_info['filename'])
        
        if os.path.exists(filepath):
            print(f"{model_name} already exists, skipping...")
            continue
            
        try:
            download_file(model_info['url'], filepath)
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")

if __name__ == "__main__":
    main()