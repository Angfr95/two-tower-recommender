import torch
import faiss
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from generate_data import load_data, encode_data
from model import TwoTowerModel

app = FastAPI(title="Two-Tower Recommender")

# Chargement au démarrage du serveur
ratings, movies, users = load_data()
ratings, user2idx, movie2idx, n_users, n_movies = encode_data(ratings)

model = TwoTowerModel(n_users, n_movies)
model.load_state_dict(torch.load('data/model.pt'))
model.eval()

index = faiss.read_index('data/faiss.index')

with open('data/mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

idx2movie = mappings['idx2movie']

@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if user_id not in user2idx:
        raise HTTPException(status_code=404, detail=f"Utilisateur {user_id} inconnu")

    user_idx = torch.tensor([user2idx[user_id]])
    with torch.no_grad():
        user_vec = model.user_tower(user_idx).numpy()

    faiss.normalize_L2(user_vec)
    scores, indices = index.search(user_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        movie_id = idx2movie[idx]
        title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        genres = movies[movies['movie_id'] == movie_id]['genres'].values[0]
        results.append({
            "movie_id": int(movie_id),
            "title": title,
            "genres": genres,
            "score": round(float(score), 4)
        })

    return {
        "user_id": user_id,
        "recommendations": results
    }

@app.get("/health")
def health():
    return {"status": "ok", "n_users": n_users, "n_movies": n_movies}