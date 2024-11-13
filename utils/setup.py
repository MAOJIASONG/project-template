import os
from pathlib import Path
import datasets
import huggingface_hub
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get cache directory from environment variable
PROJECT_CACHE_DIR = os.getenv('PROJECT_CACHE_DIR')

def setup_hf_cache():
    if not PROJECT_CACHE_DIR:
        raise ValueError("PROJECT_CACHE_DIR not set in environment variables")

    # Create cache directory if it doesn't exist
    Path(PROJECT_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Set cache for datasets
    datasets.config.HF_DATASETS_CACHE = PROJECT_CACHE_DIR
    
    # Set cache for model hub
    huggingface_hub.constants.HF_HUB_CACHE = PROJECT_CACHE_DIR
    
    # Also set environment variables (belt and suspenders approach)
    os.environ["HF_HOME"] = PROJECT_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = PROJECT_CACHE_DIR
    
    print(f"Hugging Face cache directories set to: {PROJECT_CACHE_DIR}")