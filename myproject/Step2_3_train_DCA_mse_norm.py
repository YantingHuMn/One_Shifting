import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset

import pandas as pd
import argparse
import os
import numpy as np
from itertools import product
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr, spearmanr
import json


class DCA(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim, dropout_rate=0.0):
        super(DCA, self).__init__()
        # Encoder: Linear → BN(center=True, scale=False) → ReLU → Dropout
        self.enc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn_enc1 = nn.BatchNorm1d(hidden_dim1)
        self.bn_enc1.weight.requires_grad = False  # scale=False

        self.enc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn_enc2 = nn.BatchNorm1d(hidden_dim2)
        self.bn_enc2.weight.requires_grad = False

        self.bottleneck = nn.Linear(hidden_dim2, latent_dim)
        self.bn_bottleneck = nn.BatchNorm1d(latent_dim)
        self.bn_bottleneck.weight.requires_grad = False

        # Decoder: Linear → BN(center=True, scale=False) → ReLU → Dropout
        self.dec1 = nn.Linear(latent_dim, hidden_dim2)
        self.bn_dec1 = nn.BatchNorm1d(hidden_dim2)
        self.bn_dec1.weight.requires_grad = False

        self.dec2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.bn_dec2 = nn.BatchNorm1d(hidden_dim1)
        self.bn_dec2.weight.requires_grad = False

        self.output = nn.Linear(hidden_dim1, input_dim)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def encode(self, x):
        h = F.relu(self.bn_enc1(self.enc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn_enc2(self.enc2(h)))
        h = self.dropout(h)
        z = F.relu(self.bn_bottleneck(self.bottleneck(h)))
        return z

    def decode(self, z):
        h = F.relu(self.bn_dec1(self.dec1(z)))
        h = self.dropout(h)
        h = F.relu(self.bn_dec2(self.dec2(h)))
        h = self.dropout(h)
        return F.relu(self.output(h))

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def _transformed_zero(trans):
    """
    Compute what count=0 becomes after the given transformation.
    Returns (zero_value, tolerance).
    """
    _TRANS_ZERO = {
        'no_trans':              (0.0,                    1e-8),
        'sqrt':                  (0.0,                    1e-8),
        'sqrt+1':                (np.sqrt(0 + 1),         1e-6), # = 1.0
        'sqrt+0.00001':          (np.sqrt(0.00001),       1e-8),
        'sqrt+10':               (np.sqrt(10),            1e-6),
        'sqrt+1_then_minus_1':   (np.sqrt(0 + 1) - 1,     1e-8), # = 0.0
        'log2':                  (np.log2(0 + 1),         1e-8), # = 0.0
        'log2_then_add_1':       (np.log2(0 + 1) + 1,     1e-6), # = 1.0
        'log2(count+2)':         (np.log2(0 + 2),         1e-6), # = 1.0
        'log2(count+1)+1':       (np.log2(0 + 1) + 1,     1e-6), # = 1.0
        'log(count+2)':          (np.log(0 + 2),          1e-6), # ≈ 0.693
        'count+1':               (1.0,                    1e-6),
    }
    if trans in _TRANS_ZERO:
        return _TRANS_ZERO[trans]
    return (0.0, 1e-8)


def _get_zero_nonzero_masks(x, trans):
    """Return (zero_mask, nonzero_mask) as float tensors."""
    zv, tol = _transformed_zero(trans)
    if zv == 0.0 and tol <= 1e-8:
        zero_mask = (x == 0).float()
        nonzero_mask = (x > 0).float()
    else:
        zero_mask = (torch.abs(x - zv) < tol).float()
        nonzero_mask = (x > zv + tol).float()
    return zero_mask, nonzero_mask


def weighted_mse_loss(recon_x, x, weight_strategy='fixed', zero_weight=1.0, nonzero_weight=5.0, trans=None):
    if weight_strategy == 'fixed':
        zero_mask, nonzero_mask = _get_zero_nonzero_masks(x, trans)
        weights = zero_mask * zero_weight + nonzero_mask * nonzero_weight

    elif weight_strategy == 'sparsity_aware':
        zero_mask, nonzero_mask = _get_zero_nonzero_masks(x, trans)
        sparsity = zero_mask.mean()
        zero_weight_dynamic = 1.0
        nonzero_weight_dynamic = 1.0 / (1.0 - sparsity + 1e-8)
        weights = zero_mask * zero_weight_dynamic + nonzero_mask * nonzero_weight_dynamic

    elif weight_strategy == 'magnitude':
        weights = 1.0 / (torch.abs(x) + 1.0)

    elif weight_strategy == 'focal':
        mse = (recon_x - x) ** 2
        weights = torch.pow(mse + 1e-8, 0.5)

    else:
        raise ValueError(f"Unknown weight_strategy: {weight_strategy}")

    weighted_mse = weights * (recon_x - x) ** 2
    return torch.mean(weighted_mse)


def train_one_epoch(model, loader, optimizer, device, weight_strategy='fixed',
                    zero_weight=1.0, nonzero_weight=5.0, trans=None):
    model.train()
    total = 0.0
    n = 0

    for batch_idx, (x,) in enumerate(loader):
        x = x.to(device)

        if not torch.isfinite(x).all():
            print(f"  [ERROR] NaN/Inf in input batch {batch_idx}, trans={trans}")
            return float("inf")

        optimizer.zero_grad()
        recon, z = model(x)

        if not torch.isfinite(recon).all():
            print(f"  [ERROR] NaN/Inf in reconstruction at batch {batch_idx}, trans={trans}")
            return float("inf")

        loss = weighted_mse_loss(recon, x, weight_strategy, zero_weight, nonzero_weight, trans)

        if not torch.isfinite(loss):
            print(
                f"  [ERROR] NaN/Inf training loss: {loss.item()}, "
                f"trans={trans}, zero_w={zero_weight}, nonzero_w={nonzero_weight}"
            )
            return float("inf")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += loss.item() * x.size(0)
        n += x.size(0)

    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device, weight_strategy='fixed',
              zero_weight=1.0, nonzero_weight=5.0, trans=None):
    model.eval()
    total = 0.0
    n = 0

    for batch_idx, (x,) in enumerate(loader):
        x = x.to(device)

        if not torch.isfinite(x).all():
            print(f"  [ERROR] NaN/Inf in eval input batch {batch_idx}, trans={trans}")
            return float("inf")

        recon, z = model(x)

        if not torch.isfinite(recon).all():
            print(f"  [ERROR] NaN/Inf in eval reconstruction at batch {batch_idx}, trans={trans}")
            return float("inf")

        loss = weighted_mse_loss(recon, x, weight_strategy, zero_weight, nonzero_weight, trans)

        if not torch.isfinite(loss):
            print(
                f"  [ERROR] NaN/Inf eval loss: {loss.item()}, "
                f"trans={trans}, zero_w={zero_weight}, nonzero_w={nonzero_weight}"
            )
            return float("inf")

        total += loss.item() * x.size(0)
        n += x.size(0)

    return total / max(n, 1)


@torch.no_grad()
def reconstruct_array(model, X, batch_size, device):
    model.eval()
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

    all_recon = []
    for (x,) in loader:
        x = x.to(device)
        recon, z = model(x)
        all_recon.append(recon.cpu().numpy())

    return np.vstack(all_recon)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0
        self.stopped = False
        self.best_state = None

    def step(self, metric, model=None):
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            if model is not None:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return self.stopped


def parse_grid(s, typ=int):
    return [typ(x) for x in s.split(",") if x.strip() != ""]


def should_drop_last_train(n, bs, min_bn_batch=16):
    if n <= bs:
        return False
    remainder = n % bs
    return 0 < remainder < min_bn_batch


def make_loader(X, idx, batch_size, shuffle, drop_last=False):
    subset = Subset(TensorDataset(X), idx)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


def make_file_tag(norm, trans):
    return f"norm_{norm}_trans_{trans}"


def _apply_norm(df, norm_factor):
    """
    Apply column-wise normalization to count columns only.
    The first column, usually pos/barcode, is kept unchanged.
    """
    data = df.iloc[:, 1:].astype(np.float64)
    norm_factor = str(norm_factor)

    if norm_factor == "no_norm":
        df.iloc[:, 1:] = data

    elif norm_factor == "standardize":
        col_mean = data.mean(axis=0)
        col_std = data.std(axis=0, ddof=0)
        col_std = col_std.replace(0, 1.0)
        df.iloc[:, 1:] = (data - col_mean) / col_std

    else:
        factor = float(norm_factor)
        libsize = data.sum(axis=0)
        libsize_safe = libsize.replace(0, np.nan)
        normalized = data.div(libsize_safe, axis=1) * factor
        normalized = normalized.fillna(0)
        df.iloc[:, 1:] = normalized


def _apply_trans(df, trans):
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(np.float64)

    if trans == "sqrt+1":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:] + 1)
    elif trans == "sqrt":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:])
    elif trans == "log2":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 1)
    elif trans == "log2_then_add_1":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 1) + 1
    elif trans == "sqrt+0.00001":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:] + 0.00001)
    elif trans == "sqrt+10":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:] + 10)
    elif trans == "sqrt+1_then_minus_1":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:] + 1) - 1
    elif trans == "count+1":
        df.iloc[:, 1:] = df.iloc[:, 1:] + 1
    elif trans == "log(count+2)":
        df.iloc[:, 1:] = np.log(df.iloc[:, 1:] + 2)
    elif trans == "log2(count+2)":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 2)
    elif trans == "log2(count+1)+1":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 1) + 1
    elif trans == "no_trans":
        pass
    else:
        raise ValueError(f"Unknown transformation: {trans}")


def filter_norm_transform(df, threshold_value, norm, trans):
    data_cols = df.columns[1:]
    zero_percentage = (df[data_cols] == 0).mean()
    keep_cols = zero_percentage < threshold_value

    pos_col = df.columns[0]
    cols_to_keep = [pos_col] + data_cols[keep_cols].tolist()

    filtered_raw = df[cols_to_keep].copy()
    filtered_df = df[cols_to_keep].copy()

    _apply_norm(filtered_df, norm)
    _apply_trans(filtered_df, trans)

    return filtered_raw, filtered_df


def make_recon_df_same_scale(recon_array, filtered_df):
    recon_df = pd.DataFrame(recon_array, columns=filtered_df.columns[1:])
    recon_df.insert(0, filtered_df.columns[0], filtered_df.iloc[:, 0].to_numpy())
    return recon_df


def _prepare_tensor(filtered_df):
    df = filtered_df.copy()

    if 'pos' in df.columns:
        df.drop(columns=['pos'], inplace=True)
    elif 'barcode' in df.columns:
        df.drop(columns=['barcode'], inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"[WARNING] Converting object column {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(0, inplace=True)

    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("Cannot convert to tensor: non-numeric data present")

    X = torch.tensor(df.to_numpy(), dtype=torch.float32)
    return X


def compute_input_recon_correlation(original_df, recon_df):
    common_cols = [c for c in original_df.columns[1:] if c in recon_df.columns]
    records = []

    for col in common_cols:
        x = pd.to_numeric(original_df[col], errors='coerce').fillna(0).to_numpy(dtype=np.float64)
        y = pd.to_numeric(recon_df[col], errors='coerce').fillna(0).to_numpy(dtype=np.float64)

        if np.std(x) == 0 or np.std(y) == 0:
            p = 0.0
            s = 0.0
        else:
            p, _ = pearsonr(x, y)
            s, _ = spearmanr(x, y)

            if np.isnan(p):
                p = 0.0
            if np.isnan(s):
                s = 0.0

        records.append({
            "column": col,
            "pearson": float(p),
            "spearman": float(s)
        })

    corr_df = pd.DataFrame(records)
    mean_pearson = float(corr_df["pearson"].mean()) if len(corr_df) > 0 else 0.0
    mean_spearman = float(corr_df["spearman"].mean()) if len(corr_df) > 0 else 0.0

    return mean_pearson, mean_spearman, corr_df


def rank_and_select_norm(norm_summary_df):
    df = norm_summary_df.copy()

    df["rank_pearson"] = df["mean_pearson"].rank(
        ascending=False,
        method="min"
    )

    df["rank_spearman"] = df["mean_spearman"].rank(
        ascending=False,
        method="min"
    )

    df["avg_rank"] = (df["rank_pearson"] + df["rank_spearman"]) / 2.0
    df["corr_sum"] = df["mean_pearson"] + df["mean_spearman"]

    df = df.sort_values(
        by=["avg_rank", "corr_sum"],
        ascending=[True, False]
    ).reset_index(drop=True)

    selected_norm = df.loc[0, "norm"]
    return selected_norm, df


def outer10_inner_holdout(
    df_raw, norm, device,
    hidden_grid1, hidden_grid2, latent_grid, lr_grid, bs_grid,
    dropout_grid,
    threshold_grid, trans_grid,
    epochs_inner, epochs_outer, inner_val_frac, seed,
    early_stop=False, patience=10, min_delta=0.0, check_every=1, outer_es_val_frac=0.1,
    n_splits=10, weight_strategy='fixed', zero_weight_grid=[1.0], nonzero_weight_grid=[1.0],
    save_dir=None):

    if save_dir is None:
        raise ValueError("save_dir must be provided")

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"[DATA VALIDATION] norm={norm}")
    print("=" * 60)
    print(f"  df_raw shape: {df_raw.shape}")

    df_numeric = df_raw.drop(columns=['pos'], errors='ignore').select_dtypes(include=[np.number])

    df_nan_count = df_numeric.isna().sum().sum()
    df_inf_count = np.isinf(df_numeric).sum().sum()
    df_min = df_numeric.min().min()
    df_max = df_numeric.max().max()
    df_mean = df_numeric.mean().mean()
    df_variance = df_numeric.var().mean()

    print(f"\n  df_raw statistics:")
    print(f"    NaN count: {df_nan_count}")
    print(f"    Inf count: {df_inf_count}")
    print(f"    Range: [{df_min:.4f}, {df_max:.4f}]")
    print(f"    Mean: {df_mean:.4f}")
    print(f"    Mean variance: {df_variance:.6f}")

    if df_nan_count > 0:
        print("\n  [WARNING] NaN values found in input data!")
    if df_inf_count > 0:
        print("  [WARNING] Inf values found in input data!")
    if df_variance == 0:
        print("  [WARNING] No variance in data!")
    if df_max > 1e6:
        print("  [WARNING] Very large values detected! Consider normalization.")

    print("=" * 60 + "\n")

    unsafe = {
        'sqrt', 'sqrt+1', 'sqrt+0.00001', 'sqrt+10', 'sqrt+1_then_minus_1',
        'log2', 'log2_then_add_1', 'log2(count+2)', 'log2(count+1)+1', 'log(count+2)'
    }

    df_numeric_all = df_raw.iloc[:, 1:]
    has_negatives = (df_numeric_all < 0).any().any()

    if has_negatives:
        trans_grid = [t for t in trans_grid if t not in unsafe] or ['no_trans']
        print(f"[INFO] Negative values detected, filtered trans grid to: trans={trans_grid}")

    if str(norm) == "standardize":
        if len(trans_grid) == 1 and trans_grid[0] != "no_trans":
            raise ValueError(
                f"norm=standardize is incompatible with fixed trans={trans_grid[0]}. "
                "Remove standardize from --norm_grid or use --trans_grid no_trans."
            )
        trans_grid = ["no_trans"]
        print(f"[INFO] norm=standardize uses trans=no_trans only.")

    transform_cache = {}
    tensor_cache = {}

    for threshold, trans in product(threshold_grid, trans_grid):
        cache_key = (threshold, trans)

        if cache_key not in transform_cache:
            fraw, fdf = filter_norm_transform(df_raw, threshold, norm, trans)
            transform_cache[cache_key] = (fraw, fdf)
            tensor_cache[cache_key] = _prepare_tensor(fdf)

    print(f"[CACHE] Pre-computed {len(transform_cache)} (threshold, trans) combinations for norm={norm}.\n")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_best_cfgs = []
    fold_val_losses = []
    fold_test_losses = []

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(kf.split(df_raw), 1):
        print(f"\n========== Norm {norm} | Fold {fold_id}/{n_splits} ==========")

        tr_idx, val_idx = train_test_split(
            outer_train_idx,
            test_size=inner_val_frac,
            random_state=seed,
            shuffle=True
        )

        best_cfg = None
        best_val_loss = float("inf")

        print(f"[Norm {norm} | Fold {fold_id}] Evaluating all hyperparameter combinations on validation set...")

        for threshold, trans, hidden_dim1, hidden_dim2, latent_dim, lr, bs, dropout, zero_w, nonzero_w in product(
            threshold_grid, trans_grid, hidden_grid1, hidden_grid2, latent_grid,
            lr_grid, bs_grid, dropout_grid, zero_weight_grid, nonzero_weight_grid
        ):
            cache_key = (threshold, trans)

            if cache_key not in tensor_cache:
                continue

            X = tensor_cache[cache_key]
            input_dim = X.shape[1]

            tr_loader = make_loader(
                X,
                tr_idx,
                batch_size=bs,
                shuffle=True,
                drop_last=should_drop_last_train(len(tr_idx), bs, min_bn_batch=16),
            )
            val_loader = make_loader(X, val_idx, batch_size=bs, shuffle=False, drop_last=False)

            model = DCA(input_dim, hidden_dim1, hidden_dim2, latent_dim, dropout_rate=dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            if early_stop:
                es_inner = EarlyStopping(patience=patience, min_delta=min_delta)

            for ep in range(1, epochs_inner + 1):
                train_one_epoch(
                    model, tr_loader, optimizer, device,
                    weight_strategy, zero_w, nonzero_w, trans
                )

                if early_stop and (ep % check_every == 0):
                    curr_val = eval_loss(
                        model, val_loader, device,
                        weight_strategy, zero_w, nonzero_w, trans
                    )

                    if es_inner.step(curr_val, model):
                        print(f"    [inner] early-stopped at epoch {ep}, best_val={es_inner.best:.4f}", flush=True)

                        if es_inner.best_state is not None:
                            model.load_state_dict(es_inner.best_state)
                        break

            val_loss = eval_loss(
                model, val_loader, device,
                weight_strategy, zero_w, nonzero_w, trans
            )

            if val_loss is None or not np.isfinite(val_loss):
                print(
                    f"[WARNING] Norm {norm} Fold {fold_id}: invalid val_loss={val_loss} "
                    f"for threshold={threshold}, trans={trans}, "
                    f"hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}, latent_dim={latent_dim}, "
                    f"lr={lr}, bs={bs}, dropout={dropout}, "
                    f"zero_w={zero_w}, nonzero_w={nonzero_w}"
                )
                continue

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_cfg = dict(
                    norm=norm,
                    threshold=threshold,
                    trans=trans,
                    hidden_dim1=hidden_dim1,
                    hidden_dim2=hidden_dim2,
                    latent_dim=latent_dim,
                    lr=lr,
                    batch_size=bs,
                    dropout=dropout,
                    zero_weight=zero_w,
                    nonzero_weight=nonzero_w
                )

        if best_cfg is None:
            raise ValueError(f"[Norm {norm} Fold {fold_id}] No valid config found. Check if val_loss returns NaN/inf.")

        print(f"[Norm {norm} Fold {fold_id}] Best validation config: {best_cfg}, val_loss={best_val_loss:.4f}")

        fold_best_cfgs.append(best_cfg)
        fold_val_losses.append(best_val_loss)

        print(f"[Norm {norm} Fold {fold_id}] Retraining best config on full training set and evaluating on test...")

        train_idx_full = outer_train_idx

        threshold = best_cfg["threshold"]
        trans = best_cfg["trans"]
        hidden_dim1 = best_cfg["hidden_dim1"]
        hidden_dim2 = best_cfg["hidden_dim2"]
        latent_dim = best_cfg["latent_dim"]
        lr = best_cfg["lr"]
        bs = best_cfg["batch_size"]
        dropout = best_cfg["dropout"]
        zero_w = best_cfg["zero_weight"]
        nonzero_w = best_cfg["nonzero_weight"]

        cache_key = (threshold, trans)
        X = tensor_cache[cache_key]
        input_dim = X.shape[1]

        use_outer_val = (outer_es_val_frac > 0.0)

        if use_outer_val:
            tr_full_idx, outer_val_idx = train_test_split(
                train_idx_full,
                test_size=outer_es_val_frac,
                random_state=seed,
                shuffle=True
            )
        else:
            tr_full_idx = train_idx_full

        train_loader_full = make_loader(
            X,
            tr_full_idx,
            batch_size=bs,
            shuffle=True,
            drop_last=should_drop_last_train(len(tr_full_idx), bs, min_bn_batch=16),
        )

        if use_outer_val:
            outer_val_loader = make_loader(X, outer_val_idx, batch_size=bs, shuffle=False, drop_last=False)

        model = DCA(input_dim, hidden_dim1, hidden_dim2, latent_dim, dropout_rate=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        if early_stop:
            es_outer = EarlyStopping(patience=patience, min_delta=min_delta)

        for ep in range(1, epochs_outer + 1):
            tr_loss = train_one_epoch(
                model, train_loader_full, optimizer, device,
                weight_strategy, zero_w, nonzero_w, trans
            )

            if early_stop and (ep % check_every == 0):
                if use_outer_val:
                    monitor = eval_loss(
                        model, outer_val_loader, device,
                        weight_strategy, zero_w, nonzero_w, trans
                    )
                else:
                    monitor = tr_loss

                if es_outer.step(monitor, model):
                    tag = "val" if use_outer_val else "train"
                    print(f"[outer retrain] early-stopped at epoch {ep}, best_{tag}={es_outer.best:.4f}", flush=True)

                    if es_outer.best_state is not None:
                        model.load_state_dict(es_outer.best_state)
                    break

        test_loader = make_loader(X, outer_test_idx, batch_size=bs, shuffle=False, drop_last=False)

        test_loss = eval_loss(
            model, test_loader, device,
            weight_strategy, zero_w, nonzero_w, trans
        )

        fold_test_losses.append(test_loss)

        print(f"[Test - Best Config] Test_loss={test_loss:.4f}")

    mean_test = float(np.mean(fold_test_losses))
    std_test = float(np.std(fold_test_losses, ddof=1)) if len(fold_test_losses) > 1 else 0.0

    print(f"\n[FINAL] norm={norm} {n_splits}-fold Test Loss: mean={mean_test:.4f}, sd={std_test:.4f}")

    best_fold_idx = int(np.argmin(fold_val_losses))
    final_cfg = fold_best_cfgs[best_fold_idx]

    print(f"\n[FINAL MODEL] norm={norm}, using config from best validation fold {best_fold_idx + 1}: {final_cfg}")

    threshold = final_cfg["threshold"]
    trans = final_cfg["trans"]
    hidden_dim1 = final_cfg["hidden_dim1"]
    hidden_dim2 = final_cfg["hidden_dim2"]
    latent_dim = final_cfg["latent_dim"]
    lr = final_cfg["lr"]
    bs = final_cfg["batch_size"]
    dropout = final_cfg["dropout"]
    zero_w = final_cfg["zero_weight"]
    nonzero_w = final_cfg["nonzero_weight"]

    cache_key = (threshold, trans)
    filtered_raw, filtered_df = transform_cache[cache_key]
    X = tensor_cache[cache_key]
    input_dim = X.shape[1]

    all_idx = np.arange(X.shape[0])
    train_loader_all = make_loader(
        X,
        all_idx,
        batch_size=bs,
        shuffle=True,
        drop_last=should_drop_last_train(len(all_idx), bs, min_bn_batch=16),
    )

    final_model = DCA(input_dim, hidden_dim1, hidden_dim2, latent_dim, dropout_rate=dropout).to(device)
    final_optimizer = optim.Adam(final_model.parameters(), lr=lr)

    for ep in range(1, epochs_outer + 1):
        train_one_epoch(
            final_model, train_loader_all, final_optimizer, device,
            weight_strategy, zero_w, nonzero_w, trans
        )

    final_model_config = {
        "input_dim": int(input_dim),
        "norm": str(norm),
        "threshold": float(threshold),
        "trans": str(trans),
        "hidden_dim1": int(hidden_dim1),
        "hidden_dim2": int(hidden_dim2),
        "latent_dim": int(latent_dim),
        "lr": float(lr),
        "batch_size": int(bs),
        "dropout": float(dropout),
        "zero_weight": float(zero_w),
        "nonzero_weight": float(nonzero_w),
        "weight_strategy": weight_strategy,
        "selected_from_fold": int(best_fold_idx + 1),
        "seed": int(seed)
    }

    recon_array = reconstruct_array(final_model, X, bs, device)
    recon_df_same_scale = make_recon_df_same_scale(recon_array, filtered_df)

    final_model_state_dict = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}

    return {
        "norm": norm,
        "best_cfgs_per_fold": fold_best_cfgs,
        "val_losses_per_fold": fold_val_losses,
        "test_losses_per_fold": fold_test_losses,
        "test_loss_mean": mean_test,
        "test_loss_sd": std_test,
        "final_cfg": final_cfg,
        "filtered_raw": filtered_raw,
        "filtered_df_same_scale": filtered_df,
        "recon_df_same_scale": recon_df_same_scale,
        "final_model_state_dict": final_model_state_dict,
        "final_model_config": final_model_config
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    df_raw = pd.read_feather(args.data_path)

    hidden_grid1 = parse_grid(args.hidden_grid1, int)
    hidden_grid2 = parse_grid(args.hidden_grid2, int)
    latent_grid = parse_grid(args.latent_grid, int)
    lr_grid = [float(x) for x in args.lr_grid.split(",") if x.strip() != ""]
    bs_grid = parse_grid(args.batch_size_grid, int)
    dropout_grid = [float(x) for x in args.dropout_grid.split(",") if x.strip() != ""]
    zero_weight_grid = [float(x) for x in args.zero_weight_grid.split(",") if x.strip() != ""]
    nonzero_weight_grid = [float(x) for x in args.nonzero_weight_grid.split(",") if x.strip() != ""]

    threshold_grid = [float(x) for x in args.threshold_grid.split(",") if x.strip() != ""]
    norm_grid = [x.strip() for x in args.norm_grid.split(",") if x.strip() != ""]
    trans_grid = [x.strip() for x in args.trans_grid.split(",") if x.strip() != ""]

    if len(trans_grid) != 1:
        raise ValueError(
            f"This script expects one fixed trans at a time. Got trans_grid={trans_grid}"
        )

    save_dir = os.path.join(os.path.dirname(args.out_summary), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    norm_summary_records = {}
    norm_results = {}

    for norm in norm_grid:
        try:
            results = outer10_inner_holdout(
                df_raw=df_raw,
                norm=norm,
                device=device,
                hidden_grid1=hidden_grid1,
                hidden_grid2=hidden_grid2,
                latent_grid=latent_grid,
                lr_grid=lr_grid,
                bs_grid=bs_grid,
                dropout_grid=dropout_grid,
                threshold_grid=threshold_grid,
                trans_grid=trans_grid,
                epochs_inner=args.epochs_inner,
                epochs_outer=args.epochs_outer,
                inner_val_frac=args.inner_val_frac,
                seed=args.seed,
                early_stop=args.early_stop,
                patience=args.patience,
                min_delta=args.min_delta,
                check_every=args.check_every,
                outer_es_val_frac=args.outer_es_val_frac,
                n_splits=args.n_splits,
                weight_strategy=args.weight_strategy,
                zero_weight_grid=zero_weight_grid,
                nonzero_weight_grid=nonzero_weight_grid,
                save_dir=save_dir
            )
        except Exception as e:
            print(f"[WARNING] norm={norm} failed and will be skipped. Reason: {e}", flush=True)
            continue

        if results is None:
            print(f"[WARNING] norm={norm} returned None and will be skipped.", flush=True)
            continue

        mean_pearson, mean_spearman, per_col_corr_df = compute_input_recon_correlation(
            original_df=results["filtered_df_same_scale"],
            recon_df=results["recon_df_same_scale"]
        )

        selected_trans_for_this_norm = results["final_cfg"]["trans"]

        print(
            f"[NORM CORR] norm={norm}, trans={selected_trans_for_this_norm}, "
            f"mean_pearson={mean_pearson:.6f}, mean_spearman={mean_spearman:.6f}"
        )

        norm_summary_records[norm] = {
            "norm": norm,
            "trans": selected_trans_for_this_norm,
            "mean_pearson": mean_pearson,
            "mean_spearman": mean_spearman,
            "test_loss_mean": results["test_loss_mean"],
            "test_loss_sd": results["test_loss_sd"]
        }

        norm_results[norm] = results

    if len(norm_summary_records) == 0:
        raise ValueError("No valid norm found. All norm settings failed.")

    norm_summary_df = pd.DataFrame(list(norm_summary_records.values()))

    if len(norm_grid) == 1:
        selected_norm = norm_grid[0]
        ranked_norm_df = norm_summary_df.copy()
        ranked_norm_df["rank_pearson"] = 1
        ranked_norm_df["rank_spearman"] = 1
        ranked_norm_df["avg_rank"] = 1.0
        ranked_norm_df["corr_sum"] = ranked_norm_df["mean_pearson"] + ranked_norm_df["mean_spearman"]
    else:
        selected_norm, ranked_norm_df = rank_and_select_norm(norm_summary_df)

    norm_summary_path = os.path.join(save_dir, "norm_selection_summary.csv")
    ranked_norm_df.to_csv(norm_summary_path, index=False)

    print(f"\n[SAVE] Norm selection summary saved to: {norm_summary_path}")
    print(f"[SELECTED NORM] {selected_norm}")
    print(ranked_norm_df)

    selected_result = norm_results[selected_norm]
    selected_trans = selected_result["final_cfg"]["trans"]
    selected_file_tag = make_file_tag(selected_norm, selected_trans)

    selected_weight_path = os.path.join(save_dir, f"dca_weights_{selected_file_tag}.pt")
    selected_config_path = os.path.join(save_dir, f"dca_config_{selected_file_tag}.json")

    recon_out_dir = os.path.dirname(save_dir)
    selected_input_path = os.path.join(
        recon_out_dir,
        f"input_trans_by_{selected_trans}_norm_by_{selected_norm}.feather"
    )
    selected_recon_path = os.path.join(
        recon_out_dir,
        f"reconstruct_trans_by_{selected_trans}_norm_by_{selected_norm}.feather"
    )

    torch.save(selected_result["final_model_state_dict"], selected_weight_path)

    with open(selected_config_path, "w") as f:
        json.dump(selected_result["final_model_config"], f)

    selected_result["filtered_df_same_scale"].to_feather(selected_input_path)
    selected_result["recon_df_same_scale"].to_feather(selected_recon_path)

    print(f"[SAVE] Selected final weights saved to: {selected_weight_path}")
    print(f"[SAVE] Selected final config saved to: {selected_config_path}")
    print(f"[SAVE] Selected final input saved to: {selected_input_path}")
    print(f"[SAVE] Selected final reconstruction saved to: {selected_recon_path}")

    if args.out_summary:
        out_dir = os.path.dirname(args.out_summary)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.out_summary, "w") as f:
            f.write(f"# selected_norm\t{selected_norm}\n")
            f.write(f"# selected_trans\t{selected_trans}\n")
            f.write(f"# selected_weight_path\t{selected_weight_path}\n")
            f.write(f"# selected_config_path\t{selected_config_path}\n")
            f.write(f"# selected_input_path\t{selected_input_path}\n")
            f.write(f"# selected_recon_path\t{selected_recon_path}\n")
            f.write("# norm selection rule: rank mean_pearson and mean_spearman descending; choose lowest avg_rank; if tied choose highest mean_pearson + mean_spearman\n")
            f.write("# transformation was fixed externally by --trans_grid\n")
            f.write("# normalization was selected within this fixed transformation\n")
            f.write("# selection target is dataset-specific reconstruction similarity, not out-of-sample generalization\n")
            f.write("norm\ttrans\tmean_pearson\tmean_spearman\trank_pearson\trank_spearman\tavg_rank\tcorr_sum\ttest_loss_mean\ttest_loss_sd\n")

            for _, row in ranked_norm_df.iterrows():
                f.write(
                    f"{row['norm']}\t{row['trans']}\t{row['mean_pearson']:.6f}\t{row['mean_spearman']:.6f}\t"
                    f"{row['rank_pearson']}\t{row['rank_spearman']}\t{row['avg_rank']:.6f}\t{row['corr_sum']:.6f}\t"
                    f"{row['test_loss_mean']:.6f}\t{row['test_loss_sd']:.6f}\n"
                )
        print(f"[SAVE] wrote summary to {args.out_summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-dataset DCA denoising with normalization selection on norm/trans scale"
    )

    parser.add_argument('--data_path', type=str, required=True, help='Path to feather file that needs denoising')
    parser.add_argument('--threshold_grid', type=str, default="1", help='Threshold values for filtering')
    parser.add_argument('--norm_grid', type=str, default="no_norm,1000000,100000,10000,1000,standardize", help='Normalization factors')
    parser.add_argument('--trans_grid', type=str, default="log2(count+2)", help='Transformation types')
    parser.add_argument('--hidden_grid1', type=str, default="512,1024,2048,4096") # for pbmc/multiomics data, the numbers of genes are around 22000-23000
    parser.add_argument('--hidden_grid2', type=str, default="128,256,512,1024")
    parser.add_argument('--latent_grid', type=str, default="32")
    parser.add_argument('--lr_grid', type=str, default="0.0001")
    parser.add_argument('--batch_size_grid', type=str, default="64")
    parser.add_argument('--dropout_grid', type=str, default="0.0,0.1,0.2", help='Dropout rates to search')
    parser.add_argument('--weight_strategy', type=str, default='fixed', choices=['fixed', 'sparsity_aware', 'magnitude', 'focal'])
    parser.add_argument('--zero_weight_grid', type=str, default="1.0")
    parser.add_argument('--nonzero_weight_grid', type=str, default="1.0")
    parser.add_argument('--epochs_inner', type=int, default=60)
    parser.add_argument('--epochs_outer', type=int, default=60)
    parser.add_argument('--inner_val_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--out_summary', type=str, required=True)
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--check_every', type=int, default=1)
    parser.add_argument('--outer_es_val_frac', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=5)

    args = parser.parse_args()
    main(args)
