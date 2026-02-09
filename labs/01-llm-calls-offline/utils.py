# Pré-requis :
#   - Ollama installé + lancé
#   - Modèle LLM :  ollama pull llama3:latest   (ou gemma2:2b, etc.)
#   - Embeddings : ollama pull nomic-embed-text:latest  (optionnel pour la suite)

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import requests


# =========================
# Configuration minimale
# =========================
@dataclass
class OllamaConfig:
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3:latest")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    timeout_s: float = float(os.getenv("OLLAMA_TIMEOUT", "120"))


class OllamaError(RuntimeError):
    pass


def _session(host: str) -> requests.Session:
    """
    Désactive l'usage des proxies d'entreprise/école pour localhost
    (cause fréquente de bugs en salle).
    """
    s = requests.Session()
    if "localhost" in host or "127.0.0.1" in host:
        s.trust_env = False
    return s


# =========================
# Outils pratiques
# =========================
def healthcheck(cfg: Optional[OllamaConfig] = None) -> bool:
    cfg = cfg or OllamaConfig()
    s = _session(cfg.host)
    try:
        r = s.get(f"{cfg.host}/api/tags", timeout=cfg.timeout_s)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_models(cfg: Optional[OllamaConfig] = None) -> List[str]:
    cfg = cfg or OllamaConfig()
    s = _session(cfg.host)
    try:
        r = s.get(f"{cfg.host}/api/tags", timeout=cfg.timeout_s)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", []) if "name" in m]
    except requests.RequestException as e:
        raise OllamaError(f"Impossible de lister les modèles: {e}") from e


# =========================
# Appels Ollama (bas niveau)
# =========================
def generate(
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 500,
    stream: bool = False,
    cfg: Optional[OllamaConfig] = None,
) -> Union[str, Iterable[str]]:
    cfg = cfg or OllamaConfig()
    s = _session(cfg.host)

    payload: Dict[str, Any] = {
        "model": model or cfg.model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_tokens),
        },
    }
    if system:
        payload["system"] = system

    try:
        r = s.post(
            f"{cfg.host}/api/generate",
            json=payload,
            timeout=cfg.timeout_s,
            stream=stream,
        )
        r.raise_for_status()

        if not stream:
            return r.json().get("response", "")

        def _iter() -> Iterable[str]:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                txt = chunk.get("response", "")
                if txt:
                    yield txt
                if chunk.get("done") is True:
                    break

        return _iter()

    except requests.RequestException as e:
        raise OllamaError(f"Erreur Ollama /api/generate : {e}") from e


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 500,
    stream: bool = False,
    cfg: Optional[OllamaConfig] = None,
) -> Union[str, Iterable[str]]:
    cfg = cfg or OllamaConfig()
    s = _session(cfg.host)

    payload: Dict[str, Any] = {
        "model": model or cfg.model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_tokens),
        },
    }

    try:
        r = s.post(
            f"{cfg.host}/api/chat",
            json=payload,
            timeout=cfg.timeout_s,
            stream=stream,
        )
        r.raise_for_status()

        if not stream:
            return (r.json().get("message") or {}).get("content", "")

        def _iter() -> Iterable[str]:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                msg = chunk.get("message") or {}
                txt = msg.get("content", "")
                if txt:
                    yield txt
                if chunk.get("done") is True:
                    break

        return _iter()

    except requests.RequestException as e:
        raise OllamaError(f"Erreur Ollama /api/chat : {e}") from e


def embed(
    text: str,
    model: Optional[str] = None,
    cfg: Optional[OllamaConfig] = None,
) -> List[float]:
    cfg = cfg or OllamaConfig()
    s = _session(cfg.host)

    payload = {
        "model": model or cfg.embed_model,
        "prompt": text,
    }

    try:
        r = s.post(f"{cfg.host}/api/embeddings", json=payload, timeout=cfg.timeout_s)
        r.raise_for_status()
        emb = r.json().get("embedding")
        if not isinstance(emb, list):
            raise OllamaError("Réponse embeddings invalide (champ 'embedding' absent).")
        return emb
    except requests.RequestException as e:
        raise OllamaError(f"Erreur Ollama /api/embeddings : {e}") from e


# =========================
# Wrappers compatibles notebook
# =========================
def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 500,
    model: Optional[str] = None,
):
    _ = role  # gardé pour compatibilité
    text = generate(
        prompt=prompt,
        model=model,
        temperature=temperature if temperature is not None else 0.2,
        top_p=top_p if top_p is not None else 0.9,
        max_tokens=max_tokens,
        stream=False,
    )
    return {"role": "assistant", "content": text}


def generate_with_multiple_input(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.2,
    top_p: float = 0.9,
):
    text = chat(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
    )
    return {"role": "assistant", "content": text}


if __name__ == "__main__":
    cfg = OllamaConfig()
    print("Ollama OK ?", healthcheck(cfg))
    print("Modèles (extrait):", list_models(cfg)[:8])
    out = generate_with_single_input("Explique RAG en 1 phrase.")
    print(out["content"])
