import gc
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM


MODEL_NICKNAME_TO_NAME = {
    "phi4-14B": "microsoft/phi-4",
    "mistral-24B": "mistralai/Mistral-Small-24B-Instruct-2501",
    "gemma2.5-27B": "google/gemma-2-27b-it",
    "qwen3-32B": "Qwen/Qwen3-32B",
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_model(model_nickname, seed, max_model_len=None):
    model_name = MODEL_NICKNAME_TO_NAME[model_nickname]
    print(f"Using model: {model_name}")

    llm_kwargs = {
        "model": model_name,
        "seed": seed,
        "dtype": "auto",
        "trust_remote_code": True,
        "enable_prefix_caching": True,
        "tensor_parallel_size": torch.cuda.device_count(),
        "disable_cascade_attn": True,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return llm, tokenizer


def release_model(model):
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
