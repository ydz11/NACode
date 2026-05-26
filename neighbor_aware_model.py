import torch
import torch.nn as nn

import torch
import torch.nn as nn


class NeighborAware(nn.Module):

    def __init__(
        self,
        user_emb,
        item_emb,
        user_neighbors,
        item_neighbors,
        n_users,
        n_items,

        # neighbor params
        k=5,

        # architecture params
        hidden_factor=1.0,
        num_layers=3,

        # training params
        freeze_pretrained=False,
        dropout=0.2,
    ):
        super().__init__()

        self.k = k

        emb_dim = user_emb.shape[1]

        # =====================================================
        # Debug flag
        # =====================================================

        self.debug_done = False

        # =====================================================
        # Embeddings
        # =====================================================

        self.user_emb = nn.Embedding.from_pretrained(
            user_emb,
            freeze=freeze_pretrained,
            padding_idx=0
        )

        self.item_emb = nn.Embedding.from_pretrained(
            item_emb,
            freeze=freeze_pretrained,
            padding_idx=0
        )

        # =====================================================
        # Neighbor Buffers
        # =====================================================

        user_topk = torch.zeros(
            (n_users + 1, k),
            dtype=torch.long
        )

        for u, neigh_list in user_neighbors.items():

            neigh_list = neigh_list[:k]

            user_topk[u, :len(neigh_list)] = torch.tensor(
                neigh_list,
                dtype=torch.long
            )

        item_topk = torch.zeros(
            (n_items + 1, k),
            dtype=torch.long
        )

        for i, neigh_list in item_neighbors.items():

            neigh_list = neigh_list[:k]

            item_topk[i, :len(neigh_list)] = torch.tensor(
                neigh_list,
                dtype=torch.long
            )

        self.register_buffer("user_topk_buf", user_topk)
        self.register_buffer("item_topk_buf", item_topk)

        # =====================================================
        # Input Dimension
        # =====================================================

        mlp_input_dim = 2 * (k + 1) * emb_dim

        # =====================================================
        # Hidden Dimension
        # =====================================================

        hidden_dim = max(int(mlp_input_dim * hidden_factor), 8)

        # =====================================================
        # Build MLP
        # =====================================================

        layers = []

        prev_dim = mlp_input_dim

        for layer_idx in range(num_layers):

            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # =====================================================
        # Prediction Layer
        # =====================================================

        self.predict_layer = nn.Linear(hidden_dim, 1)

        # =====================================================
        # Bias Terms
        # =====================================================

        self.user_bias = nn.Embedding(
            n_users + 1,
            1,
            padding_idx=0
        )

        self.item_bias = nn.Embedding(
            n_items + 1,
            1,
            padding_idx=0
        )

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # =====================================================
        # Architecture Info
        # =====================================================

        print("\n[NeighborAware Architecture]")
        print(f"Embedding dim : {emb_dim}")
        print(f"Neighbor k    : {k}")
        print(f"Input dim     : {mlp_input_dim}")
        print(f"Hidden dim    : {hidden_dim}")
        print(f"Num layers    : {num_layers}")
        print(f"Dropout       : {dropout}")

    def forward(self, user, item):

        # =====================================================
        # Target Embeddings
        # =====================================================

        u_target = self.user_emb(user)
        i_target = self.item_emb(item)

        # =====================================================
        # Neighbor IDs
        # =====================================================

        u_nei_ids = self.user_topk_buf[user]
        i_nei_ids = self.item_topk_buf[item]

        # =====================================================
        # Neighbor Embeddings
        # =====================================================

        u_nei_emb = self.user_emb(u_nei_ids)
        i_nei_emb = self.item_emb(i_nei_ids)

        # =====================================================
        # Mask Padding
        # =====================================================

        u_mask = u_nei_ids.eq(0).unsqueeze(-1)
        u_nei_emb = u_nei_emb.masked_fill(u_mask, 0.0)

        i_mask = i_nei_ids.eq(0).unsqueeze(-1)
        i_nei_emb = i_nei_emb.masked_fill(i_mask, 0.0)

        # =====================================================
        # Flatten
        # =====================================================

        u_nei_flat = u_nei_emb.view(
            u_target.size(0),
            -1
        )

        i_nei_flat = i_nei_emb.view(
            i_target.size(0),
            -1
        )

        # =====================================================
        # Concatenate
        # =====================================================

        u_concat = torch.cat(
            [u_target, u_nei_flat],
            dim=-1
        )

        i_concat = torch.cat(
            [i_target, i_nei_flat],
            dim=-1
        )

        mlp_input = torch.cat(
            [u_concat, i_concat],
            dim=-1
        )

        # =====================================================
        # MLP
        # =====================================================

        hidden = self.mlp(mlp_input)

        pred = self.predict_layer(hidden).squeeze(-1)

        # =====================================================
        # Add Bias
        # =====================================================

        pred = (
            pred
            + self.user_bias(user).squeeze(-1)
            + self.item_bias(item).squeeze(-1)
        )

        # =====================================================
        # Debug
        # =====================================================

        if not self.debug_done:

            print("\n[Debug NeighborAware]")
            print(f"u_nei_ids shape : {u_nei_ids.shape}")
            print(f"u_nei_ids[0]    : {u_nei_ids[0]}")
            print(f"mlp_input shape : {mlp_input.shape}")
            print(f"hidden shape    : {hidden.shape}")

            self.debug_done = True

        return pred