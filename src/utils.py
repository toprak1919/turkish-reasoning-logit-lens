import os
import random
import logging
import torch
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger(__name__)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = os.environ.get("HF_TOKEN")
        except ImportError:
            pass
    return token


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cache_results(data: dict, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(data, path)


def load_cached(path: str) -> dict:
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    return None


def format_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id])
    return repr(text) if text.strip() == "" else text


def get_gpu_info() -> str:
    if not torch.cuda.is_available():
        return "No GPU available"
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    return f"{name} ({mem_gb:.1f} GB)"
