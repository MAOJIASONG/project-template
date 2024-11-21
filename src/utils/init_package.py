import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_HF_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "huggingface")


def setup_hf_cache():
    
    cache_dir = os.getenv('PROJECT_CACHE_DIR', DEFAULT_HF_CACHE_HOME)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # os.environ["HF_HOME"] = cache_dir
    # os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    print(f"Hugging Face cache directories set to: {cache_dir}")


def setup_hydra_config():
    from hydra.core.config_store import ConfigStore
    from src.train.arguments import (
        HydraConfig
    )

    cs = ConfigStore.instance()
    cs.store(name="hydra_config", node=HydraConfig)
    # print(cs.repo)
    