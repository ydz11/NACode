import pandas as pd
import numpy as np


MAX_HISTORY_LEN = 300


def load_ratings(path):
    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(path, sep="::", engine="python", names=cols)
    df = df.sort_values(["user_id", "timestamp"]).copy()
    return df


def filter_user_interactions(df, min_user_interactions=5, max_user_interactions=100):
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[
            (user_counts >= min_user_interactions) &
            (user_counts <= max_user_interactions)
        ].index

        df = df[df["user_id"].isin(valid_users)]

    return df.reset_index(drop=True)

def reindex_ids(df):
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_map = {old: new for new, old in enumerate(unique_users, start=0)}
    item_map = {old: new for new, old in enumerate(unique_items, start=0)}

    df = df.copy()
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    return df


def get_num_users_items(df):
    n_users = int(df["user_id"].nunique())
    n_items = int(df["item_id"].nunique())
    return n_users, n_items


import pandas as pd

def print_split_stats(
    train_df,
    valid_pos_df,
    valid_neg_df,
    test_pos_df,
    test_neg_df,
    rating_threshold=4,
):
    
    def describe_df(name, df):
        if df is None or len(df) == 0:
            print(f"\n{name}: EMPTY")
            return

        n_users = df["user_id"].nunique()
        n_items = df["item_id"].nunique()
        n_interactions = len(df)

        pos_df = df[df["rating"] >= rating_threshold]
        neg_df = df[df["rating"] < rating_threshold]

        user_counts = df.groupby("user_id").size()
        pos_user_counts = pos_df.groupby("user_id").size()
        neg_user_counts = neg_df.groupby("user_id").size()

        print(f"\n===== {name} =====")
        print(f"Users                 : {n_users}")
        print(f"Items                 : {n_items}")
        print(f"Interactions          : {n_interactions}")

        print(f"Positive interactions : {len(pos_df)}")
        print(f"Negative interactions : {len(neg_df)}")

        print(f"Avg interactions/user : {user_counts.mean():.2f}")
        print(f"Median interactions   : {user_counts.median():.2f}")

        if len(pos_user_counts) > 0:
            print(f"Avg positives/user    : {pos_user_counts.mean():.2f}")

        if len(neg_user_counts) > 0:
            print(f"Avg negatives/user    : {neg_user_counts.mean():.2f}")

        print(f"Min interactions/user : {user_counts.min()}")
        print(f"Max interactions/user : {user_counts.max()}")

    # ---- merged eval sets ----
    valid_df = pd.concat([valid_pos_df, valid_neg_df], ignore_index=True)
    test_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True)

    # ---- print all ----
    describe_df("TRAIN", train_df)
    describe_df("VALID", valid_df)
    describe_df("TEST", test_df)

    # ---- extra eval info ----
    print("\n===== EVAL BREAKDOWN =====")

    print(f"Valid positives : {len(valid_pos_df)}")
    print(f"Valid negatives : {len(valid_neg_df)}")

    print(f"Test positives  : {len(test_pos_df)}")
    print(f"Test negatives  : {len(test_neg_df)}")


def ratio_split_with_eval_neg(
    df,
    train_ratio=0.70,
    valid_ratio=0.10,
    test_ratio=0.20,
    rating_threshold=3,
    min_pos_eval=2,
    seed=42,
):
    rng = np.random.default_rng(seed)

    train_rows = []
    valid_pos_rows, test_pos_rows = [], []
    valid_neg_rows, test_neg_rows = [], []

    for _, group in df.groupby("user_id"):
        group = group.sample(frac=1, random_state=seed)  # shuffle per user

        # ---- split by rating strata ----
        pos = group[group["rating"] > rating_threshold]
        neg = group[group["rating"] < rating_threshold]

        def split_part(sub_df):
            n = len(sub_df)
            if n == 0:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            idx = np.arange(n)
            rng.shuffle(idx)

            n_train = int(n * train_ratio)
            n_valid = int(n * valid_ratio)

            train_idx = idx[:n_train]
            valid_idx = idx[n_train:n_train + n_valid]
            test_idx = idx[n_train + n_valid:]

            return (
                sub_df.iloc[train_idx],
                sub_df.iloc[valid_idx],
                sub_df.iloc[test_idx],
            )

        # ---- split pos and neg separately ----
        pos_train, pos_valid, pos_test = split_part(pos)
        neg_train, neg_valid, neg_test = split_part(neg)

        # ---- combine train ----
        train_part = pd.concat([pos_train, neg_train])

        # ---- eval splits ----
        valid_pos = pos_valid
        test_pos = pos_test

        valid_neg = neg_valid
        test_neg = neg_test

        total_pos_eval = len(valid_pos) + len(test_pos)

        # ---- fallback if not enough positives ----
        if total_pos_eval < min_pos_eval:
            train_rows.append(group)
            continue

        # ---- ensure at least 1 positive per split ----
        if len(valid_pos) == 0 and len(test_pos) > 0:
            valid_pos = test_pos.iloc[:1]
            test_pos = test_pos.iloc[1:]

        if len(test_pos) == 0 and len(valid_pos) > 1:
            test_pos = valid_pos.iloc[-1:]
            valid_pos = valid_pos.iloc[:-1]

        train_rows.append(train_part)

        valid_pos_rows.append(valid_pos)
        test_pos_rows.append(test_pos)

        valid_neg_rows.append(valid_neg)
        test_neg_rows.append(test_neg)

    # ---- concat ----
    train_df = pd.concat(train_rows).reset_index(drop=True)

    valid_pos_df = pd.concat(valid_pos_rows).reset_index(drop=True) if valid_pos_rows else pd.DataFrame()
    test_pos_df = pd.concat(test_pos_rows).reset_index(drop=True) if test_pos_rows else pd.DataFrame()

    valid_neg_df = pd.concat(valid_neg_rows).reset_index(drop=True) if valid_neg_rows else pd.DataFrame()
    test_neg_df = pd.concat(test_neg_rows).reset_index(drop=True) if test_neg_rows else pd.DataFrame()
    
    
    print_split_stats(
    train_df,
    valid_pos_df,
    valid_neg_df,
    test_pos_df,
    test_neg_df,
    rating_threshold=4,
)

    return train_df, valid_pos_df, valid_neg_df, test_pos_df, test_neg_df
# def ratio_split(df, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15,
#                 rating_threshold=4, min_pos_eval=2):
#     train_rows, valid_rows, test_rows = [], [], []

#     for _, group in df.groupby("user_id"):
#         group = group.sort_values("timestamp")
#         n = len(group)
#         n_train = int(n * train_ratio)
#         n_valid = int(n * valid_ratio)

#         train_part = group.iloc[:n_train]
#         valid_part = group.iloc[n_train:n_train + n_valid]
#         test_part = group.iloc[n_train + n_valid:]

#         valid_pos = valid_part[valid_part["rating"] >= rating_threshold]
#         test_pos = test_part[test_part["rating"] >= rating_threshold]
#         total_pos_eval = len(valid_pos) + len(test_pos)

#         if total_pos_eval < min_pos_eval:
#             train_rows.append(group)
#             continue

#         if len(valid_pos) == 0:
#             valid_pos = test_pos.iloc[:1]
#             test_pos = test_pos.iloc[1:]
#         if len(test_pos) == 0:
#             test_pos = valid_pos.iloc[-1:]
#             valid_pos = valid_pos.iloc[:-1]

#         train_rows.append(train_part)
#         valid_rows.append(valid_pos)
#         test_rows.append(test_pos)

#     train_df = pd.concat(train_rows).reset_index(drop=True)
#     valid_df = pd.concat(valid_rows).reset_index(drop=True) if valid_rows else pd.DataFrame()
#     test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame()

#     return train_df, valid_df, test_df


def build_train_uir(train_df):
    return train_df[["user_id", "item_id", "rating", "timestamp"]].to_numpy(dtype=np.float64)



def build_ui(df):
    return df[["user_id", "item_id", "timestamp"]].to_numpy(dtype=np.float64)
