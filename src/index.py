import torch
import faiss
import numpy as np
import pickle
from generate_data import load_data, encode_data
from model import TwoTowerModel

def build_index():
    # Chargement des données
    ratings, movies, users = load_data()
    ratings, user2idx, movie2idx, n_users, n_movies = encode_data(ratings)

    # Chargement du modèle entraîné
    model = TwoTowerModel(n_users, n_movies)
    model.load_state_dict(torch.load('data/model.pt'))
    model.eval()

    # Génération des vecteurs pour tous les films
    all_item_idx = torch.arange(n_movies)
    with torch.no_grad():
        item_vectors = model.item_tower(all_item_idx).numpy()

    print(f"Vecteurs générés : {item_vectors.shape}")

    # Construction de l'index FAISS
    dimension = 128
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product = dot product
    faiss.normalize_L2(item_vectors)      # Normalisation pour que dot product = cosine similarity
    index.add(item_vectors)

    print(f"Index FAISS construit : {index.ntotal} vecteurs indexés")

    # Sauvegarde
    faiss.write_index(index, 'data/faiss.index')
    with open('data/mappings.pkl', 'wb') as f:
        pickle.dump({
            'user2idx': user2idx,
            'movie2idx': movie2idx,
            'idx2movie': {v: k for k, v in movie2idx.items()},
        }, f)

    print("Index et mappings sauvegardés")
    return index, model, movie2idx, movies

def recommend(user_id, index, model, movie2idx, movies, user2idx, k=10):
    # Vecteur utilisateur
    user_idx = torch.tensor([user2idx[user_id]])
    with torch.no_grad():
        user_vec = model.user_tower(user_idx).numpy()

    faiss.normalize_L2(user_vec)

    # Recherche des k films les plus proches
    scores, indices = index.search(user_vec, k)

    idx2movie = {v: k for k, v in movie2idx.items()}

    print(f"\nRecommandations pour l'utilisateur {user_id} :")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        movie_id = idx2movie[idx]
        title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {i+1}. {title} (score: {score:.4f})")

if __name__ == '__main__':
    index, model, movie2idx, movies = build_index()

    ratings, _, users = load_data()
    ratings, user2idx, movie2idx, n_users, n_movies = encode_data(ratings)

    # Test avec 3 utilisateurs différents
    recommend(1, index, model, movie2idx, movies, user2idx)
    recommend(100, index, model, movie2idx, movies, user2idx)
    recommend(500, index, model, movie2idx, movies, user2idx)