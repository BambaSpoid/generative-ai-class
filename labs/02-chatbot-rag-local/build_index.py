"""
build_index.py
----------------
Script standard pour (re)construire les embeddings et l’index RAG localement.

À lancer UNE SEULE FOIS avant d'utiliser le chatbot RAG.
"""

from pathlib import Path
import json
import numpy as np
import joblib

from utils import embed_texts, build_nn_index, load_documents

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DOCS_FILE = DATA_DIR / "docs.jsonl"
EMB_FILE = DATA_DIR / "embeddings.npy"
INDEX_FILE = DATA_DIR / "nn_index.joblib"
META_FILE = DATA_DIR / "cache_meta.json"


def main():
    print("🔧 Chargement des documents…")
    documents = load_documents(DOCS_FILE)

    texts = [doc["text"] for doc in documents]

    print(f"🧠 Calcul des embeddings ({len(texts)} documents)…")
    embeddings = embed_texts(texts)

    print("💾 Sauvegarde des embeddings")
    np.save(EMB_FILE, embeddings)

    print("📐 Construction de l’index de recherche")
    index = build_nn_index(embeddings)

    joblib.dump(index, INDEX_FILE)

    meta = {
        "nb_docs": len(texts),
        "embedding_dim": embeddings.shape[1],
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Index RAG prêt !")
    print("👉 Vous pouvez maintenant lancer le chatbot.")


if __name__ == "__main__":
    main()
