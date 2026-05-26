import numpy as np
from collections import defaultdict


MAX_HISTORY_LEN = 300


def _truncate_user_histories(train_uir, max_history_len=MAX_HISTORY_LEN):
    grouped = defaultdict(list)
    for row in train_uir:
        grouped[int(row[0])].append(row)

    kept_rows = []
    for rows in grouped.values():
        rows = sorted(rows, key=lambda x: x[3])
        kept_rows.extend(rows[-max_history_len:])
    return kept_rows


def _truncate_item_histories(rows, max_history_len=MAX_HISTORY_LEN):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row[1])].append(row)

    kept_rows = []
    for item_rows in grouped.values():
        item_rows = sorted(item_rows, key=lambda x: x[3])
        kept_rows.extend(item_rows[-max_history_len:])
    return kept_rows


def _build_sparse_histories(train_uir, max_history_len=MAX_HISTORY_LEN):
    rows = _truncate_user_histories(train_uir, max_history_len=max_history_len)
    rows = _truncate_item_histories(rows, max_history_len=max_history_len)

    user_hist = defaultdict(dict)
    item_hist = defaultdict(dict)
    for user_id, item_id, rating, _timestamp in rows:
        u = int(user_id)
        i = int(item_id)
        r = float(rating)
        user_hist[u][i] = r
        item_hist[i][u] = r
    return user_hist, item_hist

def build_matrix(entity_histories, entity_ids):
    # ---- global key space ----
    all_keys = set()
    for e in entity_ids:
        all_keys.update(entity_histories.get(e, {}).keys())

    key_to_idx = {k: i for i, k in enumerate(all_keys)}
    n = len(entity_ids)
    d = len(key_to_idx)

    X = np.zeros((n, d), dtype=np.float32)

    for i, e in enumerate(entity_ids):
        hist = entity_histories.get(e, {})
        for k, v in hist.items():
            X[i, key_to_idx[k]] = v

    return X, key_to_idx

def cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8  # avoid div-by-zero

    X_norm = X / norms
    sim_matrix = X_norm @ X_norm.T  # (N x N)

    return sim_matrix


def overlap_matrix(X):
    binary = (X > 0).astype(np.int32)
    return binary @ binary.T


def topk_from_sim(sim_matrix, overlap_matrix, entity_ids, k):
    n = sim_matrix.shape[0]

    # remove self similarity
    np.fill_diagonal(sim_matrix, -np.inf)

    # top-k indices per row
    topk_idx = np.argpartition(-sim_matrix, k, axis=1)[:, :k]

    neighbors = {}
    neighbor_sims = {}
    all_selected_sims = []
    all_selected_overlaps = []

    for i, anchor in enumerate(entity_ids):
        idxs = topk_idx[i]

        # sort only top-k
        sorted_local = idxs[np.argsort(-sim_matrix[i, idxs])]

        neighbors[anchor] = [entity_ids[j] for j in sorted_local]

        triples = [
            (entity_ids[j], float(sim_matrix[i, j]), int(overlap_matrix[i, j]))
            for j in sorted_local
        ]

        neighbor_sims[anchor] = triples

        all_selected_sims.extend([t[1] for t in triples])
        all_selected_overlaps.extend([t[2] for t in triples])

    return neighbors, neighbor_sims, all_selected_sims, all_selected_overlaps


def topk_neighbors_vectorized(entity_histories, entity_ids, k):
    print("Computing top-k neighbors (vectorized)...")

    X, _ = build_matrix(entity_histories, entity_ids)

    sim_matrix = cosine_sim_matrix(X)
    overlap_mat = overlap_matrix(X)

    return topk_from_sim(sim_matrix, overlap_mat, entity_ids, k)

def _cosine_full(a_dict, b_dict):
    if not a_dict or not b_dict:
        return 0.0, 0

    # union of keys
    all_keys = set(a_dict.keys()) | set(b_dict.keys())

    a = np.array([a_dict.get(k, 0.0) for k in all_keys], dtype=np.float32)
    b = np.array([b_dict.get(k, 0.0) for k in all_keys], dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0, 0

    # overlap still useful for diagnostics
    overlap = len(set(a_dict.keys()) & set(b_dict.keys()))

    return float(np.dot(a, b) / denom), overlap

def _topk_neighbors(entity_histories, entity_ids, k):
    neighbors = {}
    neighbor_sims = {}
    print ("Computing top-k neighbors...* 10")
    all_selected_sims = []
    all_selected_overlaps = []

    for anchor in entity_ids:
        scored = []
        anchor_hist = entity_histories.get(anchor, {})

        for other in entity_ids:
            if other == anchor:
                continue

            sim, overlap = _cosine_full(
                anchor_hist,
                entity_histories.get(other, {})
            )
            scored.append((other, sim, overlap))

        # sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        topk = scored[:k]

        neighbors[anchor] = [entity_id for entity_id, _, _ in topk]
        neighbor_sims[anchor] = topk
        

        all_selected_sims.extend([sim for _, sim, _ in topk])
        all_selected_overlaps.extend([overlap for _, _, overlap in topk])

    return neighbors, neighbor_sims, all_selected_sims, all_selected_overlaps

def _print_neighbor_diagnostics(name, neighbors, neighbor_sims, selected_sims, selected_overlaps, k):
    counts = np.array([len(v) for v in neighbors.values()], dtype=np.int64)
    zero_ratio = float(np.mean(counts == 0))
    full_ratio = float(np.mean(counts == k))

    print(f"[Neighbors] {name} avg={counts.mean():.1f}, min={counts.min()}, max={counts.max()}")
    print(f"[NeighborsDiag] {name} zero_ratio={zero_ratio:.4f}, full_k_ratio={full_ratio:.4f}")

    if selected_sims:
        sim_arr = np.array(selected_sims, dtype=np.float32)
        overlap_arr = np.array(selected_overlaps, dtype=np.int64)
        print(
            f"[NeighborsDiag] {name} sim mean={sim_arr.mean():.4f}, "
            f"min={sim_arr.min():.4f}, max={sim_arr.max():.4f}, median={np.median(sim_arr):.4f}"
        )
        print(
            f"[NeighborsDiag] {name} overlap mean={overlap_arr.mean():.2f}, "
            f"min={overlap_arr.min()}, max={overlap_arr.max()}, median={np.median(overlap_arr):.1f}"
        )

    low_count = int(np.sum(counts < k))
    print(f"[NeighborsDiag] {name} entities_with_less_than_{k}_neighbors={low_count}")

    sample_anchor = 1
    print(f"[Debug] {name.title()} {sample_anchor} neighbors: {neighbors.get(sample_anchor, [])}")
    print(f"[Debug] {name.title()} {sample_anchor} neighbor sims: {neighbor_sims.get(sample_anchor, [])}")


def build_neighbor_dicts(train_uir, n_users, n_items, k=5, 
                         max_history_len=MAX_HISTORY_LEN):
    user_hist, item_hist = _build_sparse_histories(train_uir, max_history_len=max_history_len)

    user_neighbors, user_neighbor_sims, user_selected_sims, user_selected_overlaps = topk_neighbors_vectorized(
        user_hist, range(n_users ), k
    )
    item_neighbors, item_neighbor_sims, item_selected_sims, item_selected_overlaps = topk_neighbors_vectorized(
        item_hist, range(n_items), k
        )

    _print_neighbor_diagnostics(
        "user", user_neighbors, user_neighbor_sims, user_selected_sims, user_selected_overlaps, k
    )
    _print_neighbor_diagnostics(
        "item", item_neighbors, item_neighbor_sims, item_selected_sims, item_selected_overlaps, k
    )

    return user_neighbors, item_neighbors
