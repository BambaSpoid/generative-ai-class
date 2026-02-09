"""
utils_local.py — helpers for a 100% local RAG lab using Ollama.

Assumes Ollama is running on http://localhost:11434
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


OLLAMA_URL = "http://localhost:11434"


@dataclass
class LLMResult:
    text: str
    latency_ms: int
    raw: Dict[str, Any]


def _post_json(
    url: str, payload: Dict[str, Any], timeout_s: int = 120
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


from types import SimpleNamespace


def ollama_chat(
    messages: List[Dict[str, str]],
    model: str = "llama3:latest",
    temperature: float = 0.2,
    top_p: float = 0.9,
    stream: bool = False,
    timeout_s: int = 120,
) -> LLMResult:
    """
    Chat local Ollama robuste :
    - tente /api/chat
    - si 404 (ou autre), fallback sur /api/generate
    """
    t0 = time.perf_counter()
    base = OLLAMA_URL.rstrip("/")

    # 1) Tentative /api/chat
    payload_chat = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature, "top_p": top_p},
        "stream": stream,
    }

    try:
        out = _post_json(f"{base}/api/chat", payload_chat, timeout_s=timeout_s)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = out.get("message", {}).get("content", "")
        return LLMResult(text=text, latency_ms=latency_ms, raw=out)
    except Exception:
        pass

    # 2) Fallback /api/generate
    prompt = ""
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        prompt += f"{role}:\n{content}\n\n"

    payload_gen = {
        "model": model,
        "prompt": prompt.strip(),
        "options": {"temperature": temperature, "top_p": top_p},
        "stream": stream,
    }

    out = _post_json(f"{base}/api/generate", payload_gen, timeout_s=timeout_s)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    text = out.get("response", "")
    return LLMResult(text=text, latency_ms=latency_ms, raw=out)


def ollama_embed(
    text: str,
    model: str = "nomic-embed-text",
    timeout_s: int = 120,
) -> Tuple[List[float], int]:
    """
    Call Ollama /api/embeddings.
    Returns (embedding_vector, latency_ms)
    """
    t0 = time.perf_counter()
    payload = {"model": model, "prompt": text}
    out = _post_json(f"{OLLAMA_URL}/api/embeddings", payload, timeout_s=timeout_s)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    vec = out.get("embedding", [])
    return vec, latency_ms


def now_ms() -> int:
    return int(time.time() * 1000)
