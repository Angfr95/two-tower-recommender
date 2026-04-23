import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from generate_data import load_data, encode_data, MovieLensDataset
from model import TwoTowerModel
from metrics import evaluate, print_metrics

def train():
    ratings, movies, users = load_data()
    ratings, user2idx, movie2idx, n_users, n_movies = encode_data(ratings)

    dataset = MovieLensDataset(ratings)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=1024)

    model    = TwoTowerModel(n_users, n_movies)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn  = nn.BCELoss()

    for epoch in range(5):
        model.train()
        total_loss = 0

        for batch_idx, (user_idx, item_idx, label) in enumerate(train_loader):
            optimizer.zero_grad()
            prediction = model(user_idx, item_idx)
            loss = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} | Loss moyenne : {avg_loss:.4f}")

        # Évaluation complète à chaque epoch (classification + ranking)
        metrics = evaluate(model, test_loader)
        print_metrics(metrics, title=f"Epoch {epoch+1} — Test set")

    torch.save(model.state_dict(), 'data/model.pt')
    print("Modèle sauvegardé dans data/model.pt")

if __name__ == '__main__':
    train()