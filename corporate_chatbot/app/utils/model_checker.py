import os
import requests
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

MODELS = {
    "mistral": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Matches exactly your file
    },
    "phi2": {
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi-2.Q4_K_M.gguf"  # Matches exactly your file
    },
    "bloomz": {
        "url": "https://huggingface.co/TheBloke/bloomz-560m-GGUF/resolve/main/bloomz-560m.Q4_K_M.gguf",
        "filename": "bloomz-560m.Q4_K_M.gguf"  # Matches exactly your file
    }
}

def download_model(url: str, filename: str, models_dir: str):
    filepath = os.path.join(models_dir, filename)
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        logger.error(f"Error downloading model {filename}: {str(e)}")
        return False

def check_and_download_models():
    # Get the absolute path to the models directory in the project root
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

    # Check if models exist
    missing_models = []
    for model_name, model_info in MODELS.items():
        filepath = os.path.join(models_dir, model_info['filename'])
        if not os.path.exists(filepath):
            missing_models.append((model_name, model_info))

    # If no models are missing, return True immediately
    if not missing_models:
        print("All required models are already present.")
        return True

    # Only download missing models if any
    if missing_models:
        print("\nMissing models detected. Starting download...")
        for model_name, model_info in missing_models:
            print(f"\nDownloading {model_name}...")
            success = download_model(model_info['url'], model_info['filename'], models_dir)
            if not success:
                logger.error(f"Failed to download {model_name}")
                return False

    return True