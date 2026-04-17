import torch
import torch.nn as nn

class TowerUser(nn.Module):
    def __init__(self, n_users, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(n_users, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, user_idx):
        x = self.embedding(user_idx)
        return self.network(x)


class TowerItem(nn.Module):
    def __init__(self, n_items, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(n_items, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, item_idx):
        x = self.embedding(item_idx)
        return self.network(x)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=128):
        super().__init__()
        self.user_tower = TowerUser(n_users, embedding_dim)
        self.item_tower = TowerItem(n_items, embedding_dim)

    def forward(self, user_idx, item_idx):
        user_vec = self.user_tower(user_idx)
        item_vec = self.item_tower(item_idx)
        # Dot product — mesure la proximité entre les deux vecteurs
        score = (user_vec * item_vec).sum(dim=1)
        return torch.sigmoid(score)