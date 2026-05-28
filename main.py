import json
from itertools import product as grid_product
from pathlib import Path
import pandas as pd
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import (
    build_train_uir,
    filter_user_interactions,
    get_num_users_items,
    load_ratings,
    ratio_split_with_eval_neg,
    reindex_ids,
)
from dataset import RatingTrainDataset, SasRecTrainDataset
from evaluate import build_eval_candidates, evaluate_model
from mf_model import MF
from ncf_model import NCF
from neighbor_aware_model import NeighborAware
from neighbor_retrieval import build_neighbor_dicts
from pretrain_sasrec import pretrain_sasrec
from sasrec_ncf import SASRecNCF

DATA_PATH = "ml-1m/ratings.dat"
OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 256
TRAIN_EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
SASREC_MAXLEN = 300
SASREC_EPOCHS = 40
SASREC_BATCH_SIZE = 128
TOP_K = 10
EVAL_KS = [5, 10, 20]
NUM_NEG_EVAL = 99
RATING_THRESHOLD = 4
MAX_HISTORY_LEN = 300
MIN_USER_INTERACTIONS = 10
MIN_ITEM_INTERACTIONS = 5
def train_rating_model(
    model,
    train_dataset,
    valid_rating_dataset,
    valid_df,
    valid_candidates_dict,
    model_name,
    lr=LR,
    regularization=WEIGHT_DECAY,
):
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=regularization,
    )
    loss_fn = torch.nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )

    valid_rating_loader = DataLoader(
        valid_rating_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False
    )

    for epoch in range(1, TRAIN_EPOCHS + 1):
        # -------------------
        # Training phase
        # -------------------
        model.train()
        train_mse_values = []

        for user, item, rating in train_loader:
            user = user.to(DEVICE)
            item = item.to(DEVICE)
            rating = rating.to(DEVICE).view(-1)

            pred = model(user, item).view(-1)

            mse = loss_fn(pred, rating)

            optimizer.zero_grad()
            mse.backward()

            # Gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_mse_values.append(mse.item())

        train_mse = float(np.mean(train_mse_values))

        # -------------------
        # Validation phase (MSE)
        # -------------------
        model.eval()
        val_mse_values = []

        with torch.no_grad():
            for u, i, r in valid_rating_loader:
                u = u.to(DEVICE)
                i = i.to(DEVICE)
                r = r.to(DEVICE).view(-1)

                pred = model(u, i).view(-1)
                val_mse_values.append(loss_fn(pred, r).item())

        val_mse = float(np.mean(val_mse_values))

        # -------------------
        # Ranking evaluation (monitoring only)
        # -------------------
        valid_metrics = evaluate_model(
            model,
            valid_candidates_dict,
            valid_df,
            EVAL_KS,
            DEVICE,
            debug=(epoch == 1),
        )

        main_hr = valid_metrics[TOP_K]["hr"]
        main_ndcg = valid_metrics[TOP_K]["ndcg"]

        print(
            f"[{model_name}] epoch={epoch:02d} | "
            f"train_mse={train_mse:.4f} | "
            f"valid_mse={val_mse:.4f}  |"
            f"HR@{TOP_K}={main_hr:.4f} | "
            f"NDCG@{TOP_K}={main_ndcg:.4f}"
        )

        # -------------------
        # Model selection (based on MSE)
        # -------------------
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

            if no_improve >= PATIENCE:
                print(
                    f"[{model_name}] Early stopping at epoch {epoch} "
                    f"(best_epoch={best_epoch}, best_val_mse={best_val_mse:.4f})"
                )
                break

    # -------------------
    # Restore best model
    # -------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_mse


def evaluate_rating_mse(model, rating_dataset, device):
    model.eval()
    loader = DataLoader(
        rating_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
    )
    loss_fn = torch.nn.MSELoss()
    mse_values = []

    with torch.no_grad():
        for user, item, rating in loader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device).view(-1)
            pred = model(user, item).view(-1)
            mse_values.append(loss_fn(pred, rating).item())

    return float(np.mean(mse_values)) if mse_values else 0.0


def plot_results(results, save_dir):
    
    save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    factors = sorted(results.keys())

    model_names = sorted({
        name
        for factor_results in results.values()
        for name in factor_results.keys()
    })

    # =====================================================
    # NDCG
    # =====================================================

    plt.figure(figsize=(8, 5))

    for name in model_names:

        plot_factors = [f for f in factors if name in results[f]]

        plt.plot(
            plot_factors,
            [results[f][name]["ndcg"] for f in plot_factors],
            marker="o",
            label=name
        )

    plt.xlabel("Factor")

    plt.ylabel(f"NDCG@{TOP_K}")

    plt.title("NDCG Comparison across Factors")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_dir / "ndcg_compare.png",
        dpi=200
    )

    plt.close()

    # =====================================================
    # HR
    # =====================================================

    plt.figure(figsize=(8, 5))

    for name in model_names:

        plot_factors = [f for f in factors if name in results[f]]

        plt.plot(
            plot_factors,
            [results[f][name]["hr"] for f in plot_factors],
            marker="o",
            label=name
        )

    plt.xlabel("Factor")

    plt.ylabel(f"HR@{TOP_K}")

    plt.title("HR Comparison across Factors")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_dir / "hr_compare.png",
        dpi=200
    )

    plt.close()

    # =====================================================
    # MSE
    # =====================================================

    plt.figure(figsize=(8, 5))

    for name in model_names:

        plot_factors = [f for f in factors if name in results[f]]

        plt.plot(
            plot_factors,
            [results[f][name]["mse"] for f in plot_factors],
            marker="o",
            label=name
        )

    plt.xlabel("Factor")

    plt.ylabel("Test MSE")

    plt.title("Test MSE Comparison across Factors")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_dir / "mse_compare.png",
        dpi=200
    )

    plt.close()


from itertools import product as grid_product
import json
import numpy as np
import pandas as pd


def main():

    # =========================================================
    # Setup
    # =========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # =========================================================
    # Load + Filter Data
    # =========================================================

    df = load_ratings(DATA_PATH)

    print(
        f"Raw data: {len(df)} interactions, "
        f"{df['user_id'].nunique()} users, "
        f"{df['item_id'].nunique()} items"
    )

    avg_inter = df.groupby("user_id").size().mean()
    min_inter = df.groupby("user_id").size().min()
    max_inter = df.groupby("user_id").size().max()

    print(
        f"Avg interactions per user: "
        f"{avg_inter:.1f}, min: {min_inter}, max: {max_inter}"
    )

    df = filter_user_interactions(
        df,
        min_user_interactions=MIN_USER_INTERACTIONS,
        max_user_interactions=MAX_HISTORY_LEN
    )

    print(
        f"After filtering: {len(df)} interactions, "
        f"{df['user_id'].nunique()} users, "
        f"{df['item_id'].nunique()} items"
    )

    # =========================================================
    # Reindex
    # =========================================================

    df = reindex_ids(df)

    n_users, n_items = get_num_users_items(df)

    print(f"After reindexing: n_users={n_users}, n_items={n_items}")

    # =========================================================
    # Split
    # =========================================================

    train_df, valid_pos_df, valid_neg_df, test_pos_df, test_neg_df = (
        ratio_split_with_eval_neg(
            df,
            rating_threshold=RATING_THRESHOLD
        )
    )

    print(
        f"Split: "
        f"#train={len(train_df)}, "
        f"#valid_pos={len(valid_pos_df)}, "
        f"#valid_neg={len(valid_neg_df)}, "
        f"#test_pos={len(test_pos_df)}, "
        f"#test_neg={len(test_neg_df)}"
    )

    valid_df = pd.concat(
        [valid_pos_df, valid_neg_df],
        ignore_index=True
    )

    test_df = pd.concat(
        [test_pos_df, test_neg_df],
        ignore_index=True
    )

    # =========================================================
    # Shared Objects
    # =========================================================

    train_uir = build_train_uir(train_df)

    train_dataset = RatingTrainDataset(train_df)

    valid_rating_dataset = RatingTrainDataset(valid_df)
    test_rating_dataset = RatingTrainDataset(test_df)

    user_history = {}

    for user_id, group in train_df.groupby("user_id"):

        user_history[int(user_id)] = (
            group
            .sort_values("timestamp")
            .tail(SASREC_MAXLEN)["item_id"]
            .astype(int)
            .tolist()
        )

    # =========================================================
    # Neighbor Computation
    # =========================================================

    print("\n--- Step 1: Computing neighbors ---")

    MAX_NEIGHBOR_K = 50

    user_neighbors, item_neighbors = build_neighbor_dicts(
        train_uir=train_uir,
        n_users=n_users,
        n_items=n_items,
        k=MAX_NEIGHBOR_K,
        max_history_len=MAX_HISTORY_LEN,
    )

    # =========================================================
    # Candidate Construction
    # =========================================================

    valid_uir = valid_df[
        ["user_id", "item_id", "rating"]
    ].to_numpy(dtype=np.float64)

    test_uir = test_df[
        ["user_id", "item_id", "rating"]
    ].to_numpy(dtype=np.float64)

    valid_usr_2_candidates = build_eval_candidates(
        valid_uir,
        valid_neg_df,
        num_neg=NUM_NEG_EVAL,
        seed=42
    )

    test_usr_2_candidates = build_eval_candidates(
        test_uir,
        test_neg_df,
        num_neg=NUM_NEG_EVAL,
        seed=43
    )

    # =========================================================
    # Model Specific Configs
    # =========================================================

    MODEL_CONFIGS = {

        "MF": {
            "factor": [20, 40, 60, 80, 100],
            "lr": [1e-3],
            "l2": [1e-4, 1e-3, 1e-2],
        },

        "NCF": {
            "factor": [8, 16, 32, 64],
            "num_layers": [1, 2, 3, 4],
            "dropout": [0.0],
            "lr": [1e-4, 5e-4, 1e-3, 5e-3],
            "l2": [1e-3],
        },

        "SASRec-NCF": {
            "sasrec_hidden_units": [16, 32, 64],
            "sasrec_num_neg": [1, 3, 5],
            "sasrec_lr": [1e-3, 5e-4],
            "sasrec_dropout": [0.1, 0.2, 0.4],
            "sasrec_num_blocks": [1, 2],
            "sasrec_num_heads": [1, 2],
            "num_layers": [2],
            "lr": [1e-3],
            "l2": [1e-3],
        },

       "NeighborAware": {
            "sasrec_hidden_units": [16, 32, 64],
            "neighbor_k": [5, 10, 20],
            "sasrec_num_neg": [1, 3, 5],
            "sasrec_lr": [1e-3, 5e-4],
            "sasrec_dropout": [0.1, 0.2, 0.4],
            "sasrec_num_blocks": [1, 2],
            "sasrec_num_heads": [1, 2],
            "hidden_factor": [1.0],
            "num_layers": [2],
            "dropout": [0.2, 0.9],
            "lr": [1e-3],
            "l2": [1e-3,5e-3],
        }
    }

    # =========================================================
    # SASRec Cache
    # =========================================================

    sasrec_cache = {}

    # =========================================================
    # Result Tracking
    # =========================================================

    all_tuning_results = []

    best_per_model = {}

    # =========================================================
    # Utility
    # =========================================================

    def format_multi_k(metrics, ks):

        return " | ".join(
            f"K={k}: "
            f"HR={metrics[k]['hr']:.4f}, "
            f"NDCG={metrics[k]['ndcg']:.4f}"
            for k in ks
        )

    # =========================================================
    # Main Training Loop
    # =========================================================

    for model_name, model_grid in MODEL_CONFIGS.items():

        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        grid_keys = list(model_grid.keys())
        grid_values = list(model_grid.values())

        total_configs = np.prod([len(v) for v in grid_values])

        print(f"Total configs: {total_configs}")

        for config_idx, combo in enumerate(
            grid_product(*grid_values),
            start=1
        ):

            config = dict(zip(grid_keys, combo))

            print("\n" + "-" * 80)
            print(f"[{model_name}] Config {config_idx}/{total_configs}")
            print(config)
            print("-" * 80)

            if "factor" not in config and "sasrec_hidden_units" in config:
                config["factor"] = config["sasrec_hidden_units"]

            factor = config["factor"]

            # =====================================================
            # SASRec Pretraining (ONLY if needed)
            # =====================================================

            user_emb = None
            item_emb = None

            if model_name in ["SASRec-NCF", "NeighborAware"]:

                sasrec_key = (
                    config["sasrec_hidden_units"],
                    config["sasrec_num_neg"],
                    config["sasrec_lr"],
                    config["sasrec_dropout"],
                    config["sasrec_num_blocks"],
                    config["sasrec_num_heads"],
                )

                if sasrec_key not in sasrec_cache:

                    print("\nPretraining SASRec...")
                    print(f"SASRec key: {sasrec_key}")

                    sasrec_train_dataset = SasRecTrainDataset(
                        user_history=user_history,
                        n_users=n_users,
                        n_items=n_items,
                        max_len=SASREC_MAXLEN,
                        sasrec_num_neg=config["sasrec_num_neg"],
                        seed=42,
                    )

                    user_emb, item_emb = pretrain_sasrec(
                        train_dataset=sasrec_train_dataset,
                        user_history=user_history,
                        n_users=n_users,
                        n_items=n_items,
                        device=DEVICE,
                        hidden_units=config["sasrec_hidden_units"],
                        max_len=SASREC_MAXLEN,
                        num_blocks=config["sasrec_num_blocks"],
                        num_heads=config["sasrec_num_heads"],
                        dropout_rate=config["sasrec_dropout"],
                        batch_size=SASREC_BATCH_SIZE,
                        lr=config["sasrec_lr"],
                        epochs=SASREC_EPOCHS,
                    )

                    sasrec_cache[sasrec_key] = (
                        user_emb,
                        item_emb
                    )

                else:

                    print("\nUsing cached SASRec embeddings")

                user_emb, item_emb = sasrec_cache[sasrec_key]

            # =====================================================
            # Build Model
            # =====================================================

            if model_name == "MF":

                model = MF(
                    n_users,
                    n_items,
                    embedding_dim=factor,
                )

            elif model_name == "NCF":

                model = NCF(
                    n_users,
                    n_items,
                    embedding_dim=factor,
                    hidden_dims=[factor] * config["num_layers"],
                )

            elif model_name == "SASRec-NCF":

                model = SASRecNCF(
                    user_emb,
                    item_emb,
                    hidden_dims=[factor] * config["num_layers"],
                    freeze_pretrained=True,
                )

            elif model_name == "NeighborAware":

                model = NeighborAware(
                    user_emb,
                    item_emb,
                    user_neighbors,
                    item_neighbors,
                    n_users,
                    n_items,
                    k=config["neighbor_k"],
                    num_layers=config["num_layers"],
                    freeze_pretrained=True,
                    dropout=config["dropout"],
                )

            else:
                raise ValueError(f"Unknown model: {model_name}")

            # =====================================================
            # Train
            # =====================================================

            trained_model, val_mse = train_rating_model(
                model,
                train_dataset,
                valid_rating_dataset,
                valid_df,
                valid_usr_2_candidates,
                f"{model_name}-cfg{config_idx}",
                lr=config["lr"],
                regularization=config["l2"]
            )

            # =====================================================
            # Evaluate
            # =====================================================

            valid_metrics = evaluate_model(
                trained_model,
                valid_usr_2_candidates,
                pd.concat([valid_df, test_df], ignore_index=True),
                EVAL_KS,
                DEVICE,
                debug=True
            )

            test_metrics = evaluate_model(
                trained_model,
                test_usr_2_candidates,
                pd.concat([valid_df, test_df], ignore_index=True),
                EVAL_KS,
                DEVICE,
                debug=True
            )

            # =====================================================
            # Main Metrics
            # =====================================================

            valid_hr = valid_metrics[TOP_K]["hr"]
            valid_ndcg = valid_metrics[TOP_K]["ndcg"]

            test_mse = evaluate_rating_mse(
                trained_model,
                test_rating_dataset,
                DEVICE,
            )
            test_hr = test_metrics[TOP_K]["hr"]
            test_ndcg = test_metrics[TOP_K]["ndcg"]

            # =====================================================
            # Store Results
            # =====================================================

            result_row = {
                "model": model_name,
                "config_idx": config_idx,

                **config,

                "valid_mse": val_mse,
                "test_mse": test_mse,

                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,

                "valid_hr": valid_hr,
                "valid_ndcg": valid_ndcg,

                "test_hr": test_hr,
                "test_ndcg": test_ndcg,
            }

            all_tuning_results.append(result_row)

            # =====================================================
            # Best Model Tracking
            # =====================================================

            if (
                model_name not in best_per_model
                or valid_ndcg > best_per_model[model_name]["valid_ndcg"]
            ):

                best_per_model[model_name] = result_row.copy()

            # =====================================================
            # Pretty Print
            # =====================================================

            print("\n[Validation]")
            print(f"MSE       : {val_mse:.6f}")
            print(f"HR@{TOP_K}     : {valid_hr:.4f}")
            print(f"NDCG@{TOP_K}   : {valid_ndcg:.4f}")

            print("\n[Test]")
            print(f"MSE       : {test_mse:.6f}")
            print(f"HR@{TOP_K}     : {test_hr:.4f}")
            print(f"NDCG@{TOP_K}   : {test_ndcg:.4f}")

            print("\n[Multi-K Validation]")
            print(format_multi_k(valid_metrics, EVAL_KS))

            print("\n[Multi-K Test]")
            print(format_multi_k(test_metrics, EVAL_KS))

            # =====================================================
            # Save Intermediate Results
            # =====================================================

            with open(
                OUTPUT_DIR / "gridsearch_all_results.json",
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    all_tuning_results,
                    f,
                    indent=2
                )

    # =========================================================
    # Save Final Results
    # =========================================================

    with open(
        OUTPUT_DIR / "gridsearch_all_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_tuning_results,
            f,
            indent=2
        )

    with open(
        OUTPUT_DIR / "gridsearch_best_per_model.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best_per_model,
            f,
            indent=2
        )

    # =========================================================
    # Plot Results
    # =========================================================

    all_results = {}

    for row in all_tuning_results:

        factor = row["factor"]
        model_name = row["model"]

        all_results.setdefault(factor, {})

        if (
            model_name not in all_results[factor]
            or row["valid_ndcg"]
            > all_results[factor][model_name]["ndcg"]
        ):

            all_results[factor][model_name] = {
                "hr": row["test_hr"],
                "ndcg": row["test_ndcg"],
                "mse": row["test_mse"],
            }

    with open(
        OUTPUT_DIR / "results_by_factor.json",
        "w",
        encoding="utf-8"
    ) as fp:

        json.dump(all_results, fp, indent=2)

    if all_results:
        plot_results(all_results, OUTPUT_DIR)

    # =========================================================
    # Final Summary
    # =========================================================

    print("\n" + "=" * 80)
    print("BEST CONFIG PER MODEL")
    print("=" * 80)

    for name, best in best_per_model.items():

        print(f"\n{name}")
        print("-" * 80)

        for k, v in best.items():

            if k in ["valid_metrics", "test_metrics"]:
                continue

            print(f"{k:<20}: {v}")

        print("\nDetailed Metrics")
        print(f"Valid MSE: {best['valid_mse']:.6f}")
        print(f"Test MSE : {best['test_mse']:.6f}")
        print(
            f"{'K':<6}"
            f"{'Valid HR':<12}"
            f"{'Valid NDCG':<15}"
            f"{'Test HR':<12}"
            f"{'Test NDCG':<15}"
        )

        for k in [5, 10, 20]:

            vm = best["valid_metrics"][k]
            tm = best["test_metrics"][k]

            print(
                f"{k:<6}"
                f"{vm['hr']:<12.4f}"
                f"{vm['ndcg']:<15.4f}"
                f"{tm['hr']:<12.4f}"
                f"{tm['ndcg']:<15.4f}"
            )

    print("\nGridSearch complete.")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
