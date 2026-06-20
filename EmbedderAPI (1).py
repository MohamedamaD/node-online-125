import sys
import subprocess
from functools import lru_cache
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading
import socket

# ============================================================
# INSTALL DEPENDENCIES IF MISSING
# ============================================================

def install_if_missing():
    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "torch": "torch",
        "sentence_transformers": "sentence-transformers",
        "nest_asyncio": "nest_asyncio"
    }

    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

install_if_missing()

# ============================================================
# IMPORTS AFTER INSTALL
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer
import nest_asyncio
import time

nest_asyncio.apply()

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HOST = "0.0.0.0"
PORT = 8000

# ============================================================
# MODEL LOADING (CACHED)
# ============================================================

@lru_cache(maxsize=1)
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading embedding model on {device}...")
    return SentenceTransformer(MODEL_NAME, device=device)

# ============================================================
# THREAD EXECUTOR (single worker to avoid race issues)
# ============================================================

executor = ThreadPoolExecutor(max_workers=1)

def embed_worker(text: Union[str, List[str]], normalize: bool):
    model = get_model()
    with torch.no_grad():
        return model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Medical Text Embedding API")

class EncodeRequest(BaseModel):
    text: Union[str, List[str]]
    normalize: bool = True

@app.post("/encode")
async def encode(request: EncodeRequest):
    loop = asyncio.get_running_loop()

    embeddings = await loop.run_in_executor(
        executor,
        embed_worker,
        request.text,
        request.normalize
    )

    return {
        "embeddings": embeddings.cpu().numpy().astype("float32").tolist()
    }

# ============================================================
# UTILITY: CHECK IF PORT IS OPEN
# ============================================================

def is_port_open(host: str, port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

# ============================================================
# START SERVER FUNCTION (SAFE)
# ============================================================

def start_server():
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, workers=1)

# ============================================================
# BACKGROUND START (OPTIONAL)
# ============================================================

def run_background():
    if is_port_open("127.0.0.1", PORT):
        print("✅ Embedder API already running.")
        return

    print("⚠️ Starting Embedder API in background...")
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    time.sleep(2)
    print("🚀 Embedder API started.")

# ============================================================
# ONLY RUN IF EXECUTED DIRECTLY
# ============================================================

if __name__ == "__main__":
    start_server()