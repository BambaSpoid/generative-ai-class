# 🧠 Fashion Forward Hub — Chatbot RAG 100 % Local

Ce projet implémente un **chatbot RAG (Retrieval-Augmented Generation) entièrement local**, basé sur **Ollama**, avec **API Flask**, **UI web**, **RBAC**, **observabilité** et **évaluation simple**.
Il est conçu pour un **TP de niveau Master**.

---

## ✅ Prérequis

### 1️⃣ Système

- macOS / Linux / Windows (WSL recommandé)
- **Python ≥ 3.10**

### 2️⃣ Ollama (obligatoire)

Installer Ollama :
👉 [https://ollama.com](https://ollama.com)

Vérifier que le serveur est actif :

```bash
ollama list
```

### 3️⃣ Modèles Ollama requis

```bash
ollama pull nomic-embed-text
ollama pull llama3:latest
```

_(optionnel – pour les tradeoffs)_

```bash
ollama pull gemma2:2b
ollama pull gemma3:12b
```

---

## 📦 Installation

### 1️⃣ Cloner le projet

```bash
git clone <repo-url>
cd full-rag
```

### 2️⃣ Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\\Scripts\\activate  # Windows
```

### 3️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 📁 Données & cache

Le projet utilise :

- `data/clothes_json.joblib`
- `data/faq.joblib`

⚠️ **Important**
Avant de lancer l’API, il faut **exécuter le notebook** pour :

- transformer les données en documents
- calculer les embeddings
- construire l’index vectoriel

Cela génère automatiquement :

```
data/
├── docs.jsonl
├── embeddings.npy
└── nn_index.joblib
```

---

## 🚀 Lancer l’application

### 1️⃣ Démarrer l’API Flask

```bash
python app.py
```

Tu devrais voir :

```
✅ Cache loaded: DOCS=XXXXX | EMB=(XXXXX, 768) | index=NearestNeighbors
Running on http://127.0.0.1:5000
```

---

## 🌐 Accéder aux endpoints

### 🔹 API

- **Accueil / doc rapide**
  👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

- **Health check**
  👉 [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

- **Chatbot (POST)**
  👉 [http://127.0.0.1:5000/ask](http://127.0.0.1:5000/ask)

Exemple :

```bash
curl -X POST http://127.0.0.1:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Do you have a navy blue shirt for men?","user_role":"public"}'
```

---

### 🔹 Interface Web (UI)

👉 **[http://127.0.0.1:5000/ui](http://127.0.0.1:5000/ui)**

Fonctionnalités :

- poser une question
- choisir le rôle (`public` / `staff`)
- ajuster `k` et le modèle
- voir la réponse, la latence et les documents récupérés

---

## 🧪 Vérification rapide

Dans l’UI :

1. Pose une question avec `user_role=public`
2. Repose la même avec `user_role=staff`
3. Observe la différence → **RBAC actif**

---

## 📝 Notes

- Le projet fonctionne **100 % hors ligne** (aucune clé API).
- Les réponses sont **strictement basées sur le contexte** (citations obligatoires).
- Les logs sont stockés dans `logs/traces.jsonl`.
