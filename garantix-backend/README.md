
# Garantix Backend (gratuit sur Render)

## Fichiers
- `main.py` : l'API FastAPI
- `requirements.txt` : dépendances Python

## Déploiement rapide (Render)
1) Crée un repo GitHub (ex: `garantix-backend`).
2) Ajoute ces deux fichiers (`main.py`, `requirements.txt`).
3) Sur https://render.com → **New Web Service** → **Git Provider** → choisis ton repo.
4) **Build Command**: `pip install -r requirements.txt`
5) **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6) **Environment Variables**: `OPENAI_API_KEY=sk-...`
7) Ouvre l'URL Render : `/docs` pour tester.

## Test local (facultatif)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...  # ta clé
uvicorn main:app --host 0.0.0.0 --port 8000
```
Puis http://localhost:8000/docs
