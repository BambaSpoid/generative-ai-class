# Lab CrewAI — Agent Journaliste

Objectif :

- Agent 1 : Recherche web via Serper
- Agent 2 : Rédaction d’un article
- Sortie : `outputs/nouvel-article-de-blog.md`

---

## 1) Prérequis

- Python 3.10+
- Ollama installé : https://ollama.com
- Clé API Serper (https://serper.dev)

---

## 2) Installation

```bash
cd /generativeai_class/labs/06-ai-agent
```

### Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate   # Mac
soource .venv\Scripts\activate  # Windows
```

Installer :

```bash
pip install -r requirements.txt
```

---

## 3) Configuration

Créer un fichier `.env` à la racine :

```env
SERPER_API_KEY="..."
MODEL=ollama/llama3
OLLAMA_API_BASE=http://localhost:11434
TOPIC= ----
```

---

## 4) Lancer Ollama

```bash
ollama serve
ollama pull llama3
```

Vérifier que le serveur tourne :

```bash
curl http://localhost:11434/api/tags
```

---

## 5) Exécution

```bash
python crew.py
```

Résultat :

- Article affiché dans le terminal
- Fichier généré : `outputs/nouvel-article-de-blog.md`
