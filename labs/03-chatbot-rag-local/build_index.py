"""
build_index.py
----------------
Script standard pour (re)construire les embeddings et l’index RAG localement.

À lancer UNE SEULE FOIS avant d'utiliser le chatbot RAG.

- Lit data/docs.jsonl
- Calcule les embeddings via Ollama (nomic-embed-text)
- Sauvegarde:
  - data/embeddings.npy
  - data/nn_index.joblib
  - data/cache_meta.json
"""

from pathlib import Path
import json
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

from utils import ollama_embed


DATA_DIR = Path("data")
DOCS_FILE = DATA_DIR / "docs.jsonl"
EMB_FILE = DATA_DIR / "embeddings.npy"
INDEX_FILE = DATA_DIR / "nn_index.joblib"
META_FILE = DATA_DIR / "cache_meta.json"


def load_documents(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def main():
    print("🔧 Chargement des documents…")
    documents = load_documents(DOCS_FILE)

    texts = [doc.get("text", "") for doc in documents]
    if not all(texts):
        raise ValueError("Au moins un document n'a pas de champ 'text'.")

    print(f"🧠 Calcul des embeddings ({len(texts)} documents)…")
    vectors = []
    latencies = []

    for i, text in enumerate(texts, start=1):
        vec, ms = ollama_embed(text)  # nomic-embed-text par défaut
        if not vec:
            raise RuntimeError(f"Embedding vide au doc #{i}. Vérifie Ollama + modèle.")
        vectors.append(vec)
        latencies.append(ms)

        if i % 25 == 0 or i == len(texts):
            print(f"  - {i}/{len(texts)} docs")

    embeddings = np.array(vectors, dtype=np.float32)

    print("💾 Sauvegarde des embeddings →", EMB_FILE)
    np.save(EMB_FILE, embeddings)

    print("📐 Construction de l’index NearestNeighbors")
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(embeddings)

    print("💾 Sauvegarde de l’index →", INDEX_FILE)
    joblib.dump(nn, INDEX_FILE)

    meta = {
        "nb_docs": len(texts),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_model": "nomic-embed-text",
        "avg_latency_ms": int(sum(latencies) / max(1, len(latencies))),
    }

    print("💾 Sauvegarde meta →", META_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("✅ Index RAG prêt !")
    print("👉 Vous pouvez maintenant lancer le chatbot.")


if __name__ == "__main__":
    main()
