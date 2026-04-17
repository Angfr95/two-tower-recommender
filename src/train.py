import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from generate_data import load_data, encode_data, MovieLensDataset
from model import TwoTowerModel

def train():
    # Chargement des données
    ratings, movies, users = load_data()
    ratings, user2idx, movie2idx, n_users, n_movies = encode_data(ratings)

    dataset = MovieLensDataset(ratings)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024)

    # Modèle
    model = TwoTowerModel(n_users, n_movies)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # Entraînement
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

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for user_idx, item_idx, label in test_loader:
                prediction = model(user_idx, item_idx)
                predicted_label = (prediction >= 0.5).float()
                correct += (predicted_label == label).sum().item()
                total += label.size(0)

        accuracy = correct / total
        print(f"\nEpoch {epoch+1} terminée | Loss moyenne : {avg_loss:.4f} | Accuracy : {accuracy:.4f}\n")

    # Sauvegarde
    torch.save(model.state_dict(), 'data/model.pt')
    print("Modèle sauvegardé dans data/model.pt")

if __name__ == '__main__':
    train()