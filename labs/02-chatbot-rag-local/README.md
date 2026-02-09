````markdown
# 🎓 Generative AI — Labs 100% Offline avec Ollama

Ce dépôt regroupe une série de **TP pratiques en IA Générative**, conçus pour fonctionner **entièrement hors ligne**, sans clé API ni services cloud.

---

### 1️⃣ Télécharger le projet

```bash
git clone https://github.com/BambaSpoid/generative-ai-class.git
cd generative-ai-class
```
````

_(Alternative simple : bouton **Code → Download ZIP** sur GitHub)_

---

### 2️⃣ Installer Ollama (une seule fois)

👉 [https://ollama.com](https://ollama.com)

Puis télécharger les modèles nécessaires :

```bash
ollama pull llama3:latest
ollama pull nomic-embed-text
```

---

### 3️⃣ Créer l’environnement Python

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate        # Windows
```

---

## 🧪 Contenu des Labs

### 📘 Lab 01 — Appels aux LLM & Augmentation de Prompts

📂 `labs/01-llm-calls-offline/`

**Objectifs pédagogiques :**

- Appeler un LLM depuis Python
- Comprendre la différence entre :
  - un prompt simple
  - un prompt enrichi (augmentation de contexte)

- Introduction concrète au principe des systèmes **RAG**

▶️ Lancer le TP :

```bash
cd labs/01-llm-calls-offline
pip install -r requirements.txt
jupyter notebook llm_call_ollama.ipynb
```

---

### 🤖 Lab 02 — Chatbot RAG 100% Local

📂 `labs/02-chatbot-rag-local/`

**Objectifs pédagogiques :**

- Construire un chatbot RAG local
- Comprendre :
  - embeddings
  - index de recherche
  - pipeline RAG (indexing → retrieval → génération)

- Architecture RAG **de bout en bout**, sans cloud

---

## 🔧 Étape importante — Construction de l’index RAG (obligatoire)

Avant d’utiliser le chatbot RAG ou les notebooks, il faut **construire l’index local**.

Dans le dossier du Lab 02 :

```bash
cd labs/02-chatbot-rag-local
pip install -r requirements.txt
python build_index.py
```

Ce script :

- charge les documents (`docs.jsonl`)
- calcule les embeddings avec Ollama
- construit l’index de recherche
- sauvegarde les fichiers nécessaires localement

👉 Cette étape est à faire **une seule fois**.

---

## 🖥️ Pré-requis techniques

- Python **≥ 3.10**
- Ollama (LLM local)
- Machine standard (CPU suffisant, GPU optionnel)

💡 Aucun accès Internet requis **pendant les TPs**
💡 Aucun compte, aucune clé API

---

## 📂 Structure du dépôt

```
generative-ai-class/
├── labs/
│   ├── 01-llm-calls-offline/
│   └── 02-chatbot-rag-local/
│       ├── build_index.py
│       ├── app.py
│       ├── utils.py
│       └── data/
└── README.md
```

> ⚠️ Les fichiers lourds (embeddings, index, cache) ne sont **pas versionnés**.
> Ils sont recréés automatiquement via `build_index.py`.
