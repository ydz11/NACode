import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, hidden_dims):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)

        layers = []
        input_dim = embedding_dim * 2  # concat

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user, item):
        u = self.user_embedding(user)
        i = self.item_embedding(item)

        x = torch.cat([u, i], dim=-1)
        x = self.mlp(x)
        x = self.output(x)

        return x.squeeze(-1)