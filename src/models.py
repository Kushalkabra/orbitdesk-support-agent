import time

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFIER_MODEL_ID = "facebook/bart-large-mnli"
GENERATOR_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_embedder = None
_classifier = None
_generator = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        start = time.time()
        _embedder = SentenceTransformer(EMBED_MODEL_ID)
        print(f"Loaded {EMBED_MODEL_ID} in {time.time() - start:.2f}s")
    return _embedder


def get_classifier():
    global _classifier
    if _classifier is None:
        start = time.time()
        _classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL_ID)
        print(f"Loaded {CLASSIFIER_MODEL_ID} in {time.time() - start:.2f}s")
    return _classifier


def get_generator():
    global _generator
    if _generator is None:
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_ID)
        if torch.cuda.is_available():
            # fp16 on GPU to reduce VRAM; CPU keeps default dtype
            model = AutoModelForCausalLM.from_pretrained(
                GENERATOR_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_ID)
        _generator = (model, tokenizer)
        print(f"Loaded {GENERATOR_MODEL_ID} in {time.time() - start:.2f}s")
    return _generator
