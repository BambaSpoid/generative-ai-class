## INSTALLATION

### Structure du projet

```
cv-ats/
├── app.py
├── requirements.txt
├── .env
└── venv/
```

### Créer et activer l’environnement virtuel

python -m venv venv
source venv/bin/activate # macOS / Linux

### Installer les dépendances

pip install -r requirements.txt

### Clé API Gemini

GOOGLE_API_KEY=VOTRE_CLE_API

### Lancer l’application

python -m streamlit run app.py
