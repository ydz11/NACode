import torch
import torch.nn as nn


class SASRecNCF(nn.Module):
    def __init__(self, user_emb, item_emb, hidden_dims, freeze_pretrained=False):
        super().__init__()

        n_users_plus1, emb_dim = user_emb.shape
        n_items_plus1, _ = item_emb.shape

        # --- Bug 3 fix: add padding_idx=0 to match SASRec pretraining ---
        self.user_embedding = nn.Embedding(n_users_plus1, emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items_plus1, emb_dim, padding_idx=0)

        self.user_embedding.weight.data.copy_(user_emb)
        self.item_embedding.weight.data.copy_(item_emb)

        if freeze_pretrained:
            self.user_embedding.weight.requires_grad = False
            self.item_embedding.weight.requires_grad = False

        layers = []
        input_dim = emb_dim * 2

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, user, item):
        u = self.user_embedding(user)
        i = self.item_embedding(item)

        x = torch.cat([u, i], dim=-1)
        x = self.mlp(x)
        x = self.output(x)

        return x.squeeze(-1)