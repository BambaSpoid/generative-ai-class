from __future__ import annotations

import os
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

import utils

# -----------------------------
# Config
# -----------------------------
OLLAMA_URL = "http://localhost:11434"  # utilisé indirectement par utils
EMBED_MODEL = "nomic-embed-text"
DEFAULT_MODEL = "llama3:latest"
DEFAULT_K = 8

CACHE_DIR = "data"
DOCS_PATH = os.path.join(CACHE_DIR, "docs.jsonl")
X_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
NN_PATH = os.path.join(CACHE_DIR, "nn_index.joblib")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
TRACE_FILE = LOG_DIR / "traces.jsonl"

app = Flask(__name__)

# -----------------------------
# Load cache at startup
# -----------------------------
DOCS: List[Dict[str, Any]] = []
DOC_EMB: np.ndarray | None = None
NN_INDEX: NearestNeighbors | None = None


def load_docs_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_cache() -> None:
    global DOCS, DOC_EMB, NN_INDEX
    if not (
        os.path.exists(DOCS_PATH) and os.path.exists(X_PATH) and os.path.exists(NN_PATH)
    ):
        raise FileNotFoundError(
            "Cache missing. Please run the notebook cells to build docs+embeddings+index first."
        )

    DOCS = load_docs_jsonl(DOCS_PATH)
    DOC_EMB = np.load(X_PATH).astype(np.float32, copy=False)
    NN_INDEX = joblib.load(NN_PATH)

    if len(DOCS) != DOC_EMB.shape[0]:
        raise ValueError("Cache mismatch: docs count != embeddings rows.")

    print(
        f"✅ Cache loaded: DOCS={len(DOCS)} | EMB={DOC_EMB.shape} | index={type(NN_INDEX).__name__}"
    )


def log_trace(trace: dict) -> None:
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


# -----------------------------
# RAG helpers (RBAC + retrieval + prompt)
# -----------------------------
SYSTEM = (
    "Tu es un assistant utile. "
    "Réponds UNIQUEMENT en utilisant le contexte fourni. "
    "Chaque phrase factuelle DOIT se terminer par au moins une citation entre crochets "
    "comme [prod:123] ou [faq:4]. "
    "N'invente jamais de devise (€, $, FCFA) ni de prix : réutilise exactement la valeur 'Price' du contexte. "
    "Si le contexte ne contient pas la réponse, dis exactement : "
    "Je n'ai pas assez d'informations dans les documents fournis."
)


def _rbac_allow(meta: dict, user_role: str) -> bool:
    role = (meta or {}).get("role", "public")
    return not (role == "staff" and user_role != "staff")


def build_context(hits: list, max_chars: int = 3500) -> str:
    parts, used = [], 0
    for h in hits:
        d = h["doc"]
        block = f"[{d['id']}]\n{d['text']}\n---\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts).strip()


def retrieve(query: str, k: int = 8, user_role: str = "public"):
    if NN_INDEX is None or DOC_EMB is None or len(DOCS) == 0:
        return [], 0, {"n_before_rbac": 0, "n_blocked": 0}

    qvec, qms = utils.ollama_embed(query, model=EMBED_MODEL)
    q = np.array(qvec, dtype=np.float32).reshape(1, -1)

    k = max(1, int(k))
    k0 = min(max(k * 3, k), len(DOCS))
    distances, indices = NN_INDEX.kneighbors(q, n_neighbors=k0)

    hits = []
    blocked = 0
    for dist, idx in zip(distances[0], indices[0]):
        d = DOCS[int(idx)]
        if _rbac_allow(d.get("meta", {}), user_role):
            hits.append({"doc": d, "distance": float(dist)})
        else:
            blocked += 1
        if len(hits) >= k:
            break

    return hits, int(qms), {"n_before_rbac": int(k0), "n_blocked": int(blocked)}


def build_messages(query: str, context: str) -> List[Dict[str, str]]:
    user = (
        f"Contexte :\n{context}\n\n"
        f"Question : {query}\n\n"
        "Consignes:\n"
        "- Réponds en français.\n"
        "- Chaque phrase factuelle doit contenir au moins une citation [prod:...] ou [faq:...].\n"
        "- N'invente pas d'informations absentes du contexte.\n"
        "- Si tu ne peux pas répondre avec le contexte, dis exactement :\n"
        "Je n'ai pas assez d'informations dans les documents fournis.\n"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def answer_rag(
    query: str,
    k: int = 8,
    user_role: str = "public",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> dict:
    t0 = time.perf_counter()
    hits, qms, rbac_stats = retrieve(query, k=k, user_role=user_role)
    context = build_context(hits)
    messages = build_messages(query, context)

    out = utils.ollama_chat(messages, model=model, temperature=temperature)

    latency_ms_total = int((time.perf_counter() - t0) * 1000)

    return {
        "type": "rag",
        "query": query,
        "k": int(k),
        "user_role": user_role,
        "model": model,
        "answer": out.text,
        "context": context,
        "retrieved": [
            {
                "id": h["doc"]["id"],
                "distance": h["distance"],
                "role": h["doc"].get("meta", {}).get("role", "public"),
                "type": h["doc"].get("meta", {}).get("type", "unknown"),
            }
            for h in hits
        ],
        "q_embed_ms": int(qms),
        "llm_latency_ms": int(out.latency_ms),
        "latency_ms_total": latency_ms_total,
        "rbac": rbac_stats,
        "prompt_chars": sum(len(m.get("content", "")) for m in messages),
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    try:
        # check cache loaded
        ok_cache = (NN_INDEX is not None) and (DOC_EMB is not None) and (len(DOCS) > 0)

        # check Ollama alive
        # simple ping by listing tags
        import urllib.request

        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            _ = resp.read()

        return (
            jsonify({"status": "ok", "cache_loaded": ok_cache, "docs": len(DOCS)}),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.get("/")
def index():
    return (
        jsonify(
            {
                "service": "Fashion Forward Hub — Local RAG API",
                "status": "running",
                "endpoints": {
                    "GET /health": "Check API and cache status",
                    "POST /ask": {
                        "description": "Ask a question to the RAG chatbot",
                        "payload": {
                            "query": "string (required)",
                            "user_role": "public | staff (optional, default=public)",
                            "k": "number of retrieved documents (optional, default=8)",
                            "model": "ollama model name (optional)",
                        },
                        "example": {
                            "query": "Do you have a navy blue shirt for men?",
                            "user_role": "public",
                            "k": 8,
                            "model": "llama3:latest",
                        },
                    },
                },
                "notes": [
                    "This API runs 100% locally",
                    "LLM responses are grounded in retrieved documents",
                    "Citations [prod:...] or [faq:...] are mandatory",
                ],
            }
        ),
        200,
    )


@app.get("/ask")
def ask_info():
    return (
        jsonify(
            {
                "error": "Method Not Allowed for GET on /ask",
                "how_to": "Use POST /ask with JSON payload",
                "example_curl": "curl -X POST http://127.0.0.1:5000/ask -H 'Content-Type: application/json' -d '{\"query\":\"Do you have a navy blue shirt for men?\"}'",
            }
        ),
        200,
    )


from flask import Response


@app.get("/ui")
def ui():
    html = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fashion Forward Hub — RAG Local</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; max-width: 900px; margin: 24px auto; padding: 0 16px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 12px 0; }
    label { display:block; margin: 10px 0 6px; font-weight: 600; }
    input, select, textarea, button { width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #ccc; font-size: 14px; }
    textarea { min-height: 90px; }
    button { cursor: pointer; font-weight: 700; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .muted { color: #666; font-size: 13px; }
    pre { white-space: pre-wrap; word-wrap: break-word; background: #f7f7f7; padding: 12px; border-radius: 10px; border: 1px solid #eee; }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background:#f0f0f0; margin-right: 8px; font-size: 12px; }
  </style>
</head>
<body>
  <h1>Fashion Forward Hub — Chatbot RAG (Local)</h1>
  <p class="muted">Cette page envoie une requête <span class="pill">POST /ask</span> et affiche la réponse + quelques infos (latence, documents récupérés).</p>

  <div class="card">
    <label for="query">Question</label>
    <textarea id="query" placeholder="Ex: Do you have a navy blue shirt for men?"></textarea>

    <div class="row">
      <div>
        <label for="user_role">Rôle</label>
        <select id="user_role">
          <option value="public" selected>public</option>
          <option value="staff">staff</option>
        </select>
      </div>
      <div>
        <label for="k">Top-K (documents)</label>
        <input id="k" type="number" min="1" max="20" value="8" />
      </div>
    </div>

    <div class="row">
      <div>
        <label for="model">Modèle Ollama</label>
        <input id="model" type="text" value="llama3:latest" />
      </div>
      <div>
        <label for="temperature">Température</label>
        <input id="temperature" type="number" step="0.1" min="0" max="2" value="0.2" />
      </div>
    </div>

    <button id="btn">Envoyer</button>
    <p id="status" class="muted"></p>
  </div>

  <div class="card">
    <h3>Réponse</h3>
    <pre id="answer">(vide)</pre>
  </div>

  <div class="card">
    <h3>Détails</h3>
    <p class="muted">Latence totale: <span id="lat_total">-</span> ms | LLM: <span id="lat_llm">-</span> ms | Embed: <span id="lat_emb">-</span> ms</p>
    <p class="muted">RBAC: blocked=<span id="rbac_blocked">-</span> | retrieved=<span id="n_retrieved">-</span></p>
    <pre id="retrieved">(vide)</pre>
  </div>

<script>
const $ = (id) => document.getElementById(id);

async function ask() {
  const query = $("query").value.trim();
  if (!query) {
    $("status").textContent = "⚠️ Écris une question.";
    return;
  }

  const payload = {
    query,
    user_role: $("user_role").value,
    k: Number($("k").value || 8),
    model: $("model").value.trim() || "llama3:latest",
    temperature: Number($("temperature").value || 0.2),
  };

  $("status").textContent = "⏳ Requête en cours...";
  $("answer").textContent = "(en attente)";
  $("retrieved").textContent = "(en attente)";

  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });

    const data = await resp.json();
    if (!resp.ok) {
      $("status").textContent = "❌ Erreur: " + (data.error || resp.status);
      $("answer").textContent = JSON.stringify(data, null, 2);
      return;
    }

    $("status").textContent = "✅ OK | trace_id: " + data.trace_id;

    $("answer").textContent = data.answer || "";

    $("lat_total").textContent = data.latency_ms_total ?? "-";
    $("lat_llm").textContent = data.llm_latency_ms ?? "-";
    $("lat_emb").textContent = data.q_embed_ms ?? "-";

    const blocked = data.rbac?.n_blocked ?? "-";
    $("rbac_blocked").textContent = blocked;

    const nret = (data.retrieved || []).length;
    $("n_retrieved").textContent = nret;

    // Affiche liste retrieved (ids + distance)
    const rows = (data.retrieved || []).map(x => ({
      id: x.id, distance: x.distance, role: x.role, type: x.type
    }));
    $("retrieved").textContent = JSON.stringify(rows, null, 2);

  } catch (e) {
    $("status").textContent = "❌ Exception: " + e;
  }
}

$("btn").addEventListener("click", ask);
</script>

</body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.post("/ask")
def ask():
    payload = request.get_json(force=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing field: query"}), 400

    user_role = payload.get("user_role", "public")
    k = int(payload.get("k", DEFAULT_K))
    model = payload.get("model", DEFAULT_MODEL)
    temperature = float(payload.get("temperature", 0.2))

    res = answer_rag(
        query, k=k, user_role=user_role, model=model, temperature=temperature
    )

    trace = {
        "trace_id": str(uuid.uuid4()),
        "timestamp_ms": int(time.time() * 1000),
        "query": res["query"],
        "user_role": res["user_role"],
        "k": res["k"],
        "model": res["model"],
        "n_retrieved": len(res.get("retrieved", [])),
        "retrieved_ids": [h["id"] for h in res.get("retrieved", [])],
        "rbac": res.get("rbac"),
        "llm_latency_ms": res.get("llm_latency_ms"),
        "total_latency_ms": res.get("latency_ms_total"),
        "prompt_chars": res.get("prompt_chars"),
        "answer_chars": len(res.get("answer", "")),
        "type": "rag_api",
    }
    log_trace(trace)

    # Réponse API: on renvoie sans "context" (trop lourd)
    return (
        jsonify(
            {
                "answer": res["answer"],
                "retrieved": res["retrieved"],
                "latency_ms_total": res["latency_ms_total"],
                "llm_latency_ms": res["llm_latency_ms"],
                "q_embed_ms": res["q_embed_ms"],
                "rbac": res["rbac"],
                "trace_id": trace["trace_id"],
            }
        ),
        200,
    )


if __name__ == "__main__":
    load_cache()
    # host 0.0.0.0 si tu veux exposer sur réseau local
    app.run(host="127.0.0.1", port=5000, debug=True)
