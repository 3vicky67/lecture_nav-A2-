from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .config import EMBEDDING_MODEL, DEVICE
from .db import MODEL_CACHE

tokenizer = None
model = None

def load_embedding_model() -> None:
    global tokenizer, model
    if "tokenizer" in MODEL_CACHE and "model" in MODEL_CACHE:
        tokenizer = MODEL_CACHE["tokenizer"]
        model = MODEL_CACHE["model"]
        print(f"✅ Using cached embedding model '{EMBEDDING_MODEL}'")
        return
    print(f"🚀 Loading fast embedding model '{EMBEDDING_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)
    model.eval()
    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["model"] = model
    print(f"✅ Loaded fast embedding model '{EMBEDDING_MODEL}' on {DEVICE}")

@torch.no_grad()
def embed_texts_fast(texts: List[str], batch_size: int = 16) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        with torch.cuda.amp.autocast() if DEVICE == "cuda" else torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
            arr = mean_pooled.cpu().numpy().astype("float32")
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        all_vecs.append(arr)
    return np.vstack(all_vecs)

@torch.no_grad()
def embed_query_fast(text: str) -> np.ndarray:
    return embed_texts_fast([text], batch_size=1)[0]


