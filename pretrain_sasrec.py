import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class SimpleSASRec(nn.Module):
    def __init__(self, n_items, hidden_units=64, max_len=50, num_blocks=2, num_heads=1, dropout_rate=0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden_units = hidden_units
        self.max_len = max_len

        self.item_embedding = nn.Embedding(n_items + 1, hidden_units, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.layer_norm = nn.LayerNorm(hidden_units)

    def encode(self, seq):
        device = seq.device
        positions = torch.arange(seq.size(1), device=device).unsqueeze(0).expand_as(seq)

        x = self.item_embedding(seq) + self.pos_embedding(positions)
        x = self.dropout(x)

        # padding mask
        padding_mask = seq.eq(0)

        # causal mask
        L = seq.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        h = self.layer_norm(h)
        return h

    def forward(self, seq, pos, neg):
        h = self.encode(seq)  # [B, L, H]

        pos_emb = self.item_embedding(pos)  # [B, L, H]
        neg_emb = self.item_embedding(neg)  # [B, L, sasrec_num_neg, H]

        pos_logits = (h * pos_emb).sum(dim=-1)  # [B, L]

        # h: [B, L, H] -> [B, L, 1, H] to broadcast with neg_emb
        h_expand = h.unsqueeze(2)  # [B, L, 1, H]
        neg_logits = (h_expand * neg_emb).sum(dim=-1)  # [B, L, sasrec_num_neg]

        mask = pos.ne(0).float()  # [B, L]

        pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * mask

        # Average negative loss across sasrec_num_neg negatives per position
        neg_loss_all = -torch.log(1.0 - torch.sigmoid(neg_logits) + 1e-24)  # [B, L, N]
        neg_loss = neg_loss_all.mean(dim=-1) * mask  # [B, L]

        loss = (pos_loss + neg_loss).sum() / (mask.sum() + 1e-24)
        return loss

    @torch.no_grad()
    def export_user_item_embeddings(self, user_history, n_users, device):
        self.eval()
        user_emb = torch.zeros((n_users + 1, self.hidden_units), dtype=torch.float32, device=device)

        for u in range(1, n_users + 1):
            seq_items = user_history.get(u, [])
            if len(seq_items) == 0:
                continue

            seq = np.zeros(self.max_len, dtype=np.int64)
            trunc = seq_items[-self.max_len:]
            seq[-len(trunc):] = trunc

            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            h = self.encode(seq_tensor)  # [1, L, H]

            valid_pos = (seq_tensor != 0).sum().item() - 1
            if valid_pos >= 0:
                user_emb[u] = h[0, valid_pos]

        item_emb = self.item_embedding.weight.detach().clone()
        return user_emb.cpu(), item_emb.cpu()


def pretrain_sasrec(
    train_dataset,
    user_history,
    n_users,
    n_items,
    device,
    hidden_units=64,
    max_len=50,
    num_blocks=2,
    num_heads=1,
    dropout_rate=0.2,
    batch_size=128,
    lr=1e-3,
    epochs=20,
):
    model = SimpleSASRec(
        n_items=n_items,
        hidden_units=hidden_units,
        max_len=max_len,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for _, seq, pos, neg in loader:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            loss = model(seq, pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[SASRec pretrain] epoch={epoch:02d}, loss={avg_loss:.4f}")

    user_emb, item_emb = model.export_user_item_embeddings(user_history, n_users, device)
    return user_emb, item_emb