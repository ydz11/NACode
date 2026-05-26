import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, embedding_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.user_bias = nn.Embedding(n_users + 1, 1, padding_idx=0)
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=0)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        dot = (u * i).sum(dim=-1)
        b = self.user_bias(user).squeeze(-1) + self.item_bias(item).squeeze(-1)
        return dot + b
