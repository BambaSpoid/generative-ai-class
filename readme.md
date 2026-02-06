# 📘 TP — Appels aux LLM & Augmentation de Prompts

### Étape 1 — Télécharger le TP

- Télécharger le ZIP du projet
- Décompresser le dossier

```bash
git clone https://github.com/BambaSpoid/generative-ai-class.git
cd generative-ai-class
```

## Objectifs du TP

À la fin de ce TP, l’étudiant sera capable de :

- Appeler un **modèle de langage (LLM)** depuis Python
- Utiliser un LLM avec :
  - une **question simple**
  - un **contexte enrichi** (augmentation de prompt)

- Comprendre le principe fondamental des systèmes **RAG (Retrieval-Augmented Generation)**
- Exécuter un TP **100 % hors ligne**, sans clé API ni accès cloud

Ce TP constitue une **première étape** avant l’introduction de :

- la recherche vectorielle
- les embeddings
- les RAG complets en production

---

## Pré-requis

### 1 Logiciels requis

- **Python ≥ 3.10**
- **Ollama** (serveur LLM local)

👉 Installation d’Ollama :
[https://ollama.com](https://ollama.com)

---

### 2 Modèles à installer (une seule fois)

Dans un terminal :

```bash
ollama pull llama3:latest
ollama pull nomic-embed-text
```

> 💡 `llama3` est utilisé pour la génération de texte
> 💡 `nomic-embed-text` servira plus tard pour les embeddings (RAG)

---

## Environnement Python

### 1 Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate   # windows
pip install -r requirements.txt
```

---

### 2 Installer les dépendances

```bash
pip install requests
```

_(Aucune dépendance cloud, aucun SDK propriétaire)_

---

## Lancer le TP

### 1 Vérifier qu’Ollama est lancé

```bash
ollama serve
```

Si Ollama est déjà lancé, le message suivant peut apparaître (c’est normal) :

```
address already in use
```

---

### 2 Ouvrir le notebook

```bash
jupyter notebook
```

Puis ouvrir :

```
llm_call_ollama.ipynb
```

---

## Structure du projet

```
llm-calls-rag/
├── llm_call_ollama.ipynb
├── utils.py
├── .venv/
└── README.md
```
