import numpy as np
import torch
from torch.utils.data import Dataset


class RatingTrainDataset(Dataset):
    def __init__(self, train_df):
        self.users   = train_df["user_id"].values.astype(np.int64)
        self.items   = train_df["item_id"].values.astype(np.int64)
        self.ratings = train_df["rating"].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx],   dtype=torch.long),
            torch.tensor(self.items[idx],   dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32),
        )

class SasRecTrainDataset(Dataset):
    def __init__(self, user_history, n_users, n_items,
                 max_len=50, sasrec_num_neg=1, seed=42):
        self.user_history = user_history
        self.n_users = n_users
        self.n_items = n_items
        self.max_len = max_len
        self.sasrec_num_neg = sasrec_num_neg
        self.rng = np.random.default_rng(seed)

        # Derive seen sets from user_history (no separate data structure needed)
        self.user_seen = {u: set(items) for u, items in user_history.items()}

        self.valid_users = [u for u in range(1, n_users + 1)
                            if len(user_history.get(u, [])) >= 2]

    def __len__(self):
        return len(self.valid_users)

    def __getitem__(self, idx):
        u = self.valid_users[idx]
        seq_items = self.user_history[u]

        seq = np.zeros(self.max_len, dtype=np.int64)
        pos = np.zeros(self.max_len, dtype=np.int64)
        neg = np.zeros((self.max_len, self.sasrec_num_neg), dtype=np.int64)

        nxt = seq_items[-1]
        ptr = self.max_len - 1
        seen = self.user_seen.get(u, set())

        for item in reversed(seq_items[:-1]):
            seq[ptr] = item
            pos[ptr] = nxt

            for ni in range(self.sasrec_num_neg):
                neg_item = self.rng.integers(1, self.n_items + 1)
                while neg_item in seen:
                    neg_item = self.rng.integers(1, self.n_items + 1)
                neg[ptr, ni] = neg_item

            nxt = item
            ptr -= 1
            if ptr < 0:
                break

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )