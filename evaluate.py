import numpy as np
import torch
from collections import defaultdict
import pandas as pd




def build_eval_candidates(
    eval_user_item_pairs,
    eval_neg_df,
    num_neg=100,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # ---- user → negative pool (STRICT: rating < 3) ----
    user_neg_pool = defaultdict(list)
    for row in eval_neg_df.itertuples(index=False):
        if row.rating < 4:
            user_neg_pool[int(row.user_id)].append(int(row.item_id))

    # ---- user → positive set (for safety) ----
    user_pos_set = defaultdict(set)
    for u, pos_i, _ in eval_user_item_pairs:
        user_pos_set[int(u)].add(int(pos_i))

    user_to_candidates = defaultdict(list)

    for u, pos_i, _ in eval_user_item_pairs:
        u = int(u)
        pos_i = int(pos_i)

        pool = user_neg_pool.get(u, [])
        if not pool:
            continue

        pool = np.array(pool, dtype=np.int64)

        if len(pool) == 0:
            continue

        # ---- use what exists, no filling ----
        if len(pool) > num_neg:
            sampled = rng.choice(pool, size=num_neg, replace=False)
        else:
            sampled = pool  # 🔑 use all available negatives

        # ---- build candidate list ----
        cands = np.concatenate(([pos_i], sampled))

        # ---- store per interaction (no merging) ----
        user_to_candidates[u].append((cands, 0))

    print (f"Skipped {len(eval_user_item_pairs) - len(user_to_candidates)} users with no negatives.")
    print (f"Built eval candidates for {len(user_to_candidates)} users.")
    
    return user_to_candidates



def ndcg_at_k(rel, k):
    rel = np.asarray(rel)[:k]
    if rel.size == 0:
        return 0.0

    dcg = np.sum(rel / np.log2(np.arange(2, rel.size + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, ideal.size + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, user_to_candidates, actual_df, ks, device, debug=False):
    model.eval()
    print(f"Num users in eval: {len(user_to_candidates)}")
    print(f"Example user keys: {list(user_to_candidates.keys())[:5]}")

    if isinstance(ks, int):
        ks = [ks]
    ks = sorted(set(int(k) for k in ks))
    
    # ---- build (user, item) -> rating lookup ----
    rating_lookup = {
        (int(row.user_id), int(row.item_id)): float(row.rating)
        for row in actual_df.itertuples(index=False)
    }

    user_hits = defaultdict(lambda: {k: [] for k in ks})
    user_ndcgs = defaultdict(lambda: {k: [] for k in ks})
    user_recalls = defaultdict(lambda: {k: [] for k in ks})

    users = list(user_to_candidates.keys())

    # ---- debug header ----
    if debug:
        print("\n" + "=" * 60)
        print("DEBUG: First 3 users")
        print("=" * 60)

    with torch.no_grad():
        for user_idx, u in enumerate(users):
            eval_lists = user_to_candidates[u]

            for list_idx, (cands, target_idx) in enumerate(eval_lists):
                if len(cands) == 0:
                    continue

                cands = np.asarray(cands, dtype=np.int64)

                # ---- predict ----
                u_tensor = torch.full((len(cands),), u, dtype=torch.long, device=device)
                cand_tensor = torch.tensor(cands, dtype=torch.long, device=device)

                scores = model(u_tensor, cand_tensor).view(-1).cpu().numpy()

                # ---- rank ----
                rank_order = np.argsort(-scores)

                ranked_items = cands[rank_order]
                ranked_scores = scores[rank_order]

                # ---- STRICT single-positive relevance ----
                relevance = np.zeros(len(cands), dtype=np.float64)
                relevance[target_idx] = 1.0
                ranked_rel = relevance[rank_order]

                # skip invalid (should not happen, but safe)
                if ranked_rel.sum() == 0:
                    continue

                # ---- debug ----
                if debug and user_idx < 3 and list_idx < 2:
                    print(f"\nUser {u} | List {list_idx}")
                    print("Top ranked items (item, pred_score, rel, rating):")

                    for i in range(min(10, len(ranked_items))):
                        item = int(ranked_items[i])
                        rating = rating_lookup.get((u, item), None)

                        print(
                            f"  Item {item} | "
                            f"Pred: {ranked_scores[i]:.4f} | "
                            f"Rel: {int(ranked_rel[i])} | "
                            f"Rating: {rating if rating is not None else 'NA'}"
                        )

                # ---- metrics ----
                for k in ks:
                    topk_rel = ranked_rel[:k]

                    hr = 1.0 if np.sum(topk_rel) > 0 else 0.0
                    recall = np.sum(topk_rel)  # same as HR (1 positive)
                    ndcg = ndcg_at_k(ranked_rel, k)

                    user_hits[u][k].append(hr)
                    user_recalls[u][k].append(recall)
                    user_ndcgs[u][k].append(ndcg)

    # ---- aggregate ----
    metrics = {}
    for k in ks:
        hr_vals = [np.mean(v[k]) for v in user_hits.values() if len(v[k]) > 0]
        recall_vals = [np.mean(v[k]) for v in user_recalls.values() if len(v[k]) > 0]
        ndcg_vals = [np.mean(v[k]) for v in user_ndcgs.values() if len(v[k]) > 0]

        metrics[k] = {
            "hr": float(np.mean(hr_vals)) if hr_vals else 0.0,
            "recall": float(np.mean(recall_vals)) if recall_vals else 0.0,
            "ndcg": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        }

    return metrics