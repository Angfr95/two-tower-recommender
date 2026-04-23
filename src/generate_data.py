import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

def load_data():
    ratings = pd.read_csv(
        'data/ml-1m/ratings.dat',
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    movies = pd.read_csv(
        'data/ml-1m/movies.dat',
        sep='::',
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    users = pd.read_csv(
        'data/ml-1m/users.dat',
        sep='::',
        names=['user_id', 'gender', 'age', 'occupation', 'zip'],
        engine='python'
    )
    return ratings, movies, users

def encode_data(ratings):
    # Convertit les IDs en indices continus 0, 1, 2...
    user_ids = ratings['user_id'].unique()
    movie_ids = ratings['movie_id'].unique()

    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    ratings['user_idx'] = ratings['user_id'].map(user2idx)
    ratings['movie_idx'] = ratings['movie_id'].map(movie2idx)

    # Un like = rating >= 4, sinon 0
    ratings['label'] = (ratings['rating'] >= 4).astype(float)

    n_users = len(user_ids)
    n_movies = len(movie_ids)

    print(f"Utilisateurs uniques : {n_users}")
    print(f"Films uniques        : {n_movies}")
    print(f"Interactions positives : {ratings['label'].sum():.0f}")

    return ratings, user2idx, movie2idx, n_users, n_movies

class MovieLensDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = torch.tensor(ratings_df['user_idx'].values, dtype=torch.long)
        self.movies = torch.tensor(ratings_df['movie_idx'].values, dtype=torch.long)
        self.labels = torch.tensor(ratings_df['label'].values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]

if __name__ == '__main__':
    ratings, movies, users = load_data()
    ratings, user2idx, movie2idx, n_users, n_movies = encode_data(ratings)

    dataset = MovieLensDataset(ratings)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    print(f"\nTrain : {len(train_set)} exemples")
    print(f"Test  : {len(test_set)} exemples")
    print(f"\nExemple batch : {dataset[0]}")