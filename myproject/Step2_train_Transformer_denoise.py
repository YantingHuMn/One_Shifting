import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset

import pandas as pd
import argparse
import os
import numpy as np
import math
from itertools import product
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr, spearmanr
import json
from pathlib import Path


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based autoencoder: V1 -> V1 denoising.

    Strategy for high-dimensional gene data (10k-20k features):
    1. Split input into chunks (pseudo-tokens)
    2. Project each chunk to d_model dimensions
    3. Add positional encoding
    4. Apply Transformer encoder layers (self-attention)
    5. Project back to input dimension (same as input)
    """
    def __init__(self, input_dim, n_tokens=64, d_model=128,
                 nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.n_tokens = n_tokens
        self.d_model = d_model

        # Chunk size: how many input features per token
        self.chunk_size = math.ceil(input_dim / n_tokens)
        # Pad input to be evenly divisible
        self.padded_input_dim = self.chunk_size * n_tokens

        # Input projection: each chunk -> d_model
        self.input_proj = nn.Linear(self.chunk_size, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection: flatten transformer output -> input_dim
        self.output_proj = nn.Sequential(
            nn.Linear(n_tokens * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
            nn.ReLU()  # Non-negative outputs for count data
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.size(0)

        # Pad input if needed
        if self.input_dim < self.padded_input_dim:
            padding = torch.zeros(batch_size, self.padded_input_dim - self.input_dim,
                                  device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)

        # Reshape into tokens: (batch, n_tokens, chunk_size)
        x = x.view(batch_size, self.n_tokens, self.chunk_size)

        # Project each chunk to d_model
        x = self.input_proj(x)  # (batch, n_tokens, d_model)

        # Add positional encoding
        x = x + self.pos_embedding

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, n_tokens, d_model)

        # Flatten and project to output
        x = x.reshape(batch_size, -1)  # (batch, n_tokens * d_model)
        out = self.output_proj(x)  # (batch, input_dim)

        return out


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
    total_loss = 0.0
    n = 0
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon = model(x)
        loss = weighted_mse_loss(recon, x, weight_strategy, zero_weight, nonzero_weight, trans)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device, weight_strategy='fixed',
              zero_weight=1.0, nonzero_weight=5.0, trans=None):
    model.eval()
    total_loss = 0.0
    n = 0
    for (x,) in loader:
        x = x.to(device)
        recon = model(x)
        loss = weighted_mse_loss(recon, x, weight_strategy, zero_weight, nonzero_weight, trans)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_correlation_residual(model, loader, v2_data, indices, device, corr_type='pearson'):
    model.eval()
    all_v1 = []
    all_recon = []

    for (x,) in loader:
        x = x.to(device)
        recon = model(x)
        all_v1.append(x.cpu().numpy())
        all_recon.append(recon.cpu().numpy())

    v1_array = np.vstack(all_v1)
    recon_array = np.vstack(all_recon)
    v2_subset = v2_data[indices]

    n_cols = v1_array.shape[1]
    residuals = []

    for col_idx in range(n_cols):
        v1_col = v1_array[:, col_idx]
        recon_col = recon_array[:, col_idx]
        v2_col = v2_subset[:, col_idx]

        if np.std(v1_col) == 0 or np.std(v2_col) == 0:
            corr_v1_v2 = 0.0
        else:
            if corr_type == 'pearson':
                corr_v1_v2, _ = pearsonr(v1_col, v2_col)
            else:
                corr_v1_v2, _ = spearmanr(v1_col, v2_col)
            if np.isnan(corr_v1_v2):
                corr_v1_v2 = 0.0

        if np.any(np.isnan(recon_col)) or np.std(recon_col) == 0:
            corr_recon_v2 = 0.0
        else:
            if corr_type == 'pearson':
                corr_recon_v2, _ = pearsonr(recon_col, v2_col)
            else:
                corr_recon_v2, _ = spearmanr(recon_col, v2_col)
            if np.isnan(corr_recon_v2):
                corr_recon_v2 = 0.0

        residuals.append(corr_v1_v2 - corr_recon_v2)

    return np.mean(residuals) if residuals else 0.0


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


def make_loader(X, idx, batch_size, shuffle):
    subset = Subset(TensorDataset(X), idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def _apply_trans(df, trans):
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
    elif trans == "log2(count+2)":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 2)
    elif trans == "log2(count+1)+1":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 1) + 1
    elif trans == "no_trans":
        pass


def filter_and_transform(df1, df2, threshold_value, trans1, trans2, data_path1=None, data_path2=None, save=False):
    data_cols = df1.columns[1:]
    zero_percentage = (df1[data_cols] == 0).mean()
    keep_cols = zero_percentage < threshold_value

    pos_col = df1.columns[0]
    cols_to_keep = [pos_col] + data_cols[keep_cols].tolist()

    filtered_df1 = df1[cols_to_keep].copy()
    filtered_df2 = df2[cols_to_keep].copy()

    _apply_trans(filtered_df1, trans1)
    _apply_trans(filtered_df2, trans2)

    if save:
        if data_path1 is not None:
            save_dir1 = Path(data_path1).parent
            save_dir1.mkdir(parents=True, exist_ok=True)
            save_path1 = save_dir1 / f"Count_matrix_transformed_{trans1}.feather"
            filtered_df1.to_feather(save_path1)
            print(f"[SAVE] Saved transformed df1 to: {save_path1}")

        if data_path2 is not None:
            save_dir2 = Path(data_path2).parent
            save_dir2.mkdir(parents=True, exist_ok=True)
            save_path2 = save_dir2 / f"Count_matrix_transformed_{trans2}.feather"
            filtered_df2.to_feather(save_path2)
            print(f"[SAVE] Saved transformed df2 to: {save_path2}")

    return filtered_df1, filtered_df2


def _prepare_tensors(filtered_df1, filtered_df2):
    """Drop pos/barcode columns, validate dtypes, return (X_tensor, X2_numpy)."""
    df1 = filtered_df1.copy()
    df2 = filtered_df2.copy()

    for df in [df1, df2]:
        if 'pos' in df.columns:
            df.drop(columns=['pos'], inplace=True)
        elif 'barcode' in df.columns:
            df.drop(columns=['barcode'], inplace=True)

    for df, name in [(df1, 'df1'), (df2, 'df2')]:
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"[WARNING] Converting object column {col} in {name}")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)

        if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError(f"Cannot convert to tensor: non-numeric data present in {name}")

    X = torch.tensor(df1.to_numpy(), dtype=torch.float32)
    X2 = df2.to_numpy()
    return X, X2


def outer_cv_inner_holdout(
    df1_raw, df2_raw, device,
    n_tokens_grid, d_model_grid, nhead_grid, num_layers_grid,
    dim_feedforward_grid, dropout_grid,
    lr_grid, bs_grid,
    threshold_grid, trans1_grid, trans2_grid,
    epochs_inner, epochs_outer, inner_val_frac, seed,
    early_stop=False, patience=10, min_delta=0.0, check_every=1, outer_es_val_frac=0.1,
    n_splits=10, weight_strategy='fixed', zero_weight_grid=[1.0], nonzero_weight_grid=[5.0],
    save_dir=None, eval_metric='val_loss'):

    if save_dir is None:
        raise ValueError("save_dir must be provided")
    os.makedirs(save_dir, exist_ok=True)

    # --- Negative-value safety check ---
    df1_numeric = df1_raw.iloc[:, 1:]
    has_negatives_df1 = (df1_numeric < 0).any().any()
    df2_numeric = df2_raw.iloc[:, 1:]
    has_negatives_df2 = (df2_numeric < 0).any().any()
    if has_negatives_df1 or has_negatives_df2:
        unsafe = {'sqrt', 'sqrt+1', 'sqrt+0.00001', 'sqrt+10', 'sqrt+1_then_minus_1',
                  'log2', 'log2_then_add_1', 'log2(count+2)', 'log2(count+1)+1', 'log(count+2)'}
        if has_negatives_df1:
            trans1_grid = [t for t in trans1_grid if t not in unsafe] or ['no_trans']
        if has_negatives_df2:
            trans2_grid = [t for t in trans2_grid if t not in unsafe] or ['no_trans']
        print(f"[INFO] Negative values detected (df1={has_negatives_df1}, df2={has_negatives_df2}), "
              f"filtered trans grids to: trans1={trans1_grid}, trans2={trans2_grid}")

    # --- Pre-compute all (threshold, trans1, trans2) combinations ---
    transform_cache = {}
    tensor_cache = {}
    for threshold, trans1, trans2 in product(threshold_grid, trans1_grid, trans2_grid):
        cache_key = (threshold, trans1, trans2)
        if cache_key not in transform_cache:
            fdf1, fdf2 = filter_and_transform(df1_raw, df2_raw, threshold, trans1, trans2)
            transform_cache[cache_key] = (fdf1, fdf2)
            tensor_cache[cache_key] = _prepare_tensors(fdf1, fdf2)
    print(f"[CACHE] Pre-computed {len(transform_cache)} (threshold, trans1, trans2) combinations.\n")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_best_cfgs = []
    fold_val_losses = []
    fold_test_losses = []
    fold_val_metrics = []
    fold_test_metrics = []

    all_val_results = []

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(kf.split(df1_raw), 1):
        print(f"\n========== Fold {fold_id}/{n_splits} ==========")
        tr_idx, val_idx = train_test_split(outer_train_idx, test_size=inner_val_frac,
                                           random_state=seed, shuffle=True)

        best_cfg = None
        best_val = float("inf")
        best_val_loss = float("inf")

        fold_val_combinations = []

        # ---- First pass: evaluate all configs on validation set ----
        print(f"[Fold {fold_id}] Evaluating hyperparameter combinations on validation set...")
        for threshold, trans1, trans2, n_tokens, d_model, nhead, num_layers, dim_ff, dropout, lr, bs, zero_w, nonzero_w in product(
            threshold_grid, trans1_grid, trans2_grid, n_tokens_grid, d_model_grid, nhead_grid, num_layers_grid,
            dim_feedforward_grid, dropout_grid, lr_grid, bs_grid,
            zero_weight_grid, nonzero_weight_grid):

            if d_model % nhead != 0:
                continue

            cache_key = (threshold, trans1, trans2)
            X, X2 = tensor_cache[cache_key]
            input_dim = X.shape[1]

            tr_loader = make_loader(X, tr_idx, batch_size=bs, shuffle=True)
            val_loader = make_loader(X, val_idx, batch_size=bs, shuffle=False)

            model = TransformerAutoencoder(
                input_dim=input_dim, n_tokens=n_tokens, d_model=d_model,
                nhead=nhead, num_layers=num_layers,
                dim_feedforward=dim_ff, dropout=dropout
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            if early_stop:
                es_inner = EarlyStopping(patience=patience, min_delta=min_delta)
            for ep in range(1, epochs_inner + 1):
                train_one_epoch(model, tr_loader, optimizer, device,
                                weight_strategy, zero_w, nonzero_w, trans1)
                if early_stop and (ep % check_every == 0):
                    if eval_metric == 'val_loss':
                        curr_val = eval_loss(model, val_loader, device,
                                             weight_strategy, zero_w, nonzero_w, trans1)
                    elif eval_metric in ['pearson', 'spearman']:
                        curr_val = eval_correlation_residual(model, val_loader,
                                                             X2, val_idx, device, eval_metric)
                    else:
                        raise ValueError(f"Unknown eval_metric: {eval_metric}")

                    if es_inner.step(curr_val, model):
                        print(f"    [inner] early-stopped at epoch {ep}, best_val={es_inner.best:.4f}", flush=True)
                        if es_inner.best_state is not None:
                            model.load_state_dict(es_inner.best_state)
                        break

            if eval_metric == 'val_loss':
                val_metric = eval_loss(model, val_loader, device,
                                       weight_strategy, zero_w, nonzero_w, trans1)
                val_loss = val_metric
            elif eval_metric in ['pearson', 'spearman']:
                val_metric = eval_correlation_residual(model, val_loader,
                                                       X2, val_idx, device, eval_metric)
                val_loss = eval_loss(model, val_loader, device,
                                     weight_strategy, zero_w, nonzero_w, trans1)
            else:
                raise ValueError(f"Unknown eval_metric: {eval_metric}")

            config_name = f"{zero_w}_{nonzero_w}_{trans1}_{trans2}_{threshold}_{n_tokens}_{d_model}_{nhead}_{num_layers}_{dropout}"
            fold_val_combinations.append({
                'config_name': config_name,
                'val_metric': val_metric,
                'fold': fold_id
            })

            if val_metric < best_val:
                best_val = val_metric
                best_val_loss = val_loss
                best_cfg = dict(threshold=threshold, trans1=trans1, trans2=trans2,
                                n_tokens=n_tokens, d_model=d_model,
                                nhead=nhead, num_layers=num_layers,
                                dim_feedforward=dim_ff, dropout=dropout,
                                lr=lr, batch_size=bs,
                                zero_weight=zero_w, nonzero_weight=nonzero_w)

        all_val_results.extend(fold_val_combinations)
        if best_cfg is None:
            raise ValueError(f"[Fold {fold_id}] No valid config found.")

        print(f"[Fold {fold_id}] Best config: {best_cfg}, val_metric={best_val:.4f}")

        fold_best_cfgs.append(best_cfg)
        fold_val_losses.append(best_val_loss)
        fold_val_metrics.append(best_val)

        # ---- Second pass: retrain ONLY best_cfg on full training set ----
        print(f"[Fold {fold_id}] Retraining best config on full training set...")
        train_idx_full = outer_train_idx

        threshold = best_cfg["threshold"]
        trans1 = best_cfg["trans1"]
        trans2 = best_cfg["trans2"]
        n_tokens = best_cfg["n_tokens"]
        d_model = best_cfg["d_model"]
        nhead = best_cfg["nhead"]
        num_layers = best_cfg["num_layers"]
        dim_ff = best_cfg["dim_feedforward"]
        dropout = best_cfg["dropout"]
        lr = best_cfg["lr"]
        bs = best_cfg["batch_size"]
        zero_w = best_cfg["zero_weight"]
        nonzero_w = best_cfg["nonzero_weight"]

        cache_key = (threshold, trans1, trans2)
        X, X2 = tensor_cache[cache_key]
        input_dim = X.shape[1]

        use_outer_val = (outer_es_val_frac > 0.0)
        if use_outer_val:
            tr_full_idx, outer_val_idx = train_test_split(train_idx_full, test_size=outer_es_val_frac,
                                                           random_state=seed, shuffle=True)
        else:
            tr_full_idx = train_idx_full

        train_loader_full = make_loader(X, tr_full_idx, batch_size=bs, shuffle=True)
        if use_outer_val:
            outer_val_loader = make_loader(X, outer_val_idx, batch_size=bs, shuffle=False)

        model = TransformerAutoencoder(
            input_dim=input_dim, n_tokens=n_tokens, d_model=d_model,
            nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        if early_stop:
            es_outer = EarlyStopping(patience=patience, min_delta=min_delta)
        for ep in range(1, epochs_outer + 1):
            tr_loss = train_one_epoch(model, train_loader_full, optimizer, device,
                                      weight_strategy, zero_w, nonzero_w, trans1)
            if early_stop and (ep % check_every == 0):
                if use_outer_val:
                    if eval_metric == 'val_loss':
                        monitor = eval_loss(model, outer_val_loader, device,
                                            weight_strategy, zero_w, nonzero_w, trans1)
                    elif eval_metric in ['pearson', 'spearman']:
                        monitor = eval_correlation_residual(model, outer_val_loader,
                                                            X2, outer_val_idx, device, eval_metric)
                else:
                    monitor = tr_loss

                if es_outer.step(monitor, model):
                    tag = "val" if use_outer_val else "train"
                    print(f"[outer retrain] early-stopped at epoch {ep}, best_{tag}={es_outer.best:.4f}", flush=True)
                    if es_outer.best_state is not None:
                        model.load_state_dict(es_outer.best_state)
                    break

        # Evaluate on test set
        test_loader = make_loader(X, outer_test_idx, batch_size=bs, shuffle=False)
        test_loss = eval_loss(model, test_loader, device,
                              weight_strategy, zero_w, nonzero_w, trans1)

        if eval_metric == 'val_loss':
            test_metric = test_loss
        elif eval_metric in ['pearson', 'spearman']:
            test_metric = eval_correlation_residual(model, test_loader,
                                                    X2, outer_test_idx, device, eval_metric)
        else:
            test_metric = test_loss

        fold_test_losses.append(test_loss)
        fold_test_metrics.append(test_metric)

        # Save model
        fold_dir = os.path.join(save_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, "transformer_ae_weights.pt"))

        with open(os.path.join(fold_dir, "transformer_ae_config.json"), "w") as f:
            json.dump({
                "model_type": "TransformerAutoencoder",
                "input_dim": int(input_dim),
                "threshold": float(best_cfg["threshold"]),
                "trans1": str(best_cfg["trans1"]),
                "trans2": str(best_cfg["trans2"]),
                "n_tokens": int(best_cfg["n_tokens"]),
                "d_model": int(best_cfg["d_model"]),
                "nhead": int(best_cfg["nhead"]),
                "num_layers": int(best_cfg["num_layers"]),
                "dim_feedforward": int(best_cfg["dim_feedforward"]),
                "dropout": float(best_cfg["dropout"]),
                "lr": float(best_cfg["lr"]),
                "batch_size": int(best_cfg["batch_size"]),
                "zero_weight": float(best_cfg["zero_weight"]),
                "nonzero_weight": float(best_cfg["nonzero_weight"]),
                "weight_strategy": weight_strategy,
                "eval_metric": eval_metric,
                "inner_val_loss": float(best_val_loss),
                "inner_val_metric": float(best_val),
                "outer_test_loss": float(test_loss),
                "outer_test_metric": float(test_metric),
                "seed": int(seed)
            }, f)
        print(f"[Test - Best Config] test_loss={test_loss:.4f}, test_metric={test_metric:.4f}")
        print(f"[SAVE] saved to: {fold_dir}")

    # Save validation summary CSV
    val_df = pd.DataFrame(all_val_results)
    val_df_grouped = val_df.groupby('config_name')['val_metric'].mean().reset_index()
    val_df_grouped.columns = ['config_name', 'mean_val_metric']
    val_df_grouped = val_df_grouped.sort_values('mean_val_metric')
    val_path = os.path.join(save_dir, 'all_validation_results.csv')
    val_df_grouped.to_csv(val_path, index=False)
    print(f"\n[SAVE] All validation results saved to: {val_path}")

    print(f"\n{'='*60}")
    print("TOP 5 CONFIGURATIONS BY VALIDATION METRIC:")
    print(val_df_grouped.head())

    mean_test = float(np.mean(fold_test_losses))
    std_test = float(np.std(fold_test_losses, ddof=1)) if len(fold_test_losses) > 1 else 0.0
    mean_test_metric = float(np.mean(fold_test_metrics))
    std_test_metric = float(np.std(fold_test_metrics, ddof=1)) if len(fold_test_metrics) > 1 else 0.0

    print(f"\n[FINAL] {n_splits}-fold Test Loss: mean={mean_test:.4f}, sd={std_test:.4f}")
    print(f"[FINAL] {n_splits}-fold Test Metric ({eval_metric}): mean={mean_test_metric:.4f}, sd={std_test_metric:.4f}")

    return {
        "best_cfgs_per_fold": fold_best_cfgs,
        "val_losses_per_fold": fold_val_losses,
        "val_metrics_per_fold": fold_val_metrics,
        "test_losses_per_fold": fold_test_losses,
        "test_metrics_per_fold": fold_test_metrics,
        "test_loss_mean": mean_test,
        "test_loss_sd": std_test,
        "test_metric_mean": mean_test_metric,
        "test_metric_sd": std_test_metric,
        "eval_metric": eval_metric,
        "validation_df": val_df_grouped
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    df1_raw = pd.read_feather(args.data_path1)
    df2_raw = pd.read_feather(args.data_path2)

    n_tokens_grid = parse_grid(args.n_tokens_grid, int)
    d_model_grid = parse_grid(args.d_model_grid, int)
    nhead_grid = parse_grid(args.nhead_grid, int)
    num_layers_grid = parse_grid(args.num_layers_grid, int)
    dim_feedforward_grid = parse_grid(args.dim_feedforward_grid, int)
    dropout_grid = [float(x) for x in args.dropout_grid.split(",") if x.strip() != ""]
    lr_grid = [float(x) for x in args.lr_grid.split(",") if x.strip() != ""]
    bs_grid = parse_grid(args.batch_size_grid, int)
    zero_weight_grid = [float(x) for x in args.zero_weight_grid.split(",") if x.strip() != ""]
    nonzero_weight_grid = [float(x) for x in args.nonzero_weight_grid.split(",") if x.strip() != ""]

    threshold_grid = [float(x) for x in args.threshold_grid.split(",") if x.strip() != ""]
    trans1_grid = [x.strip() for x in args.trans1_grid.split(",") if x.strip() != ""]
    trans2_grid = [x.strip() for x in args.trans2_grid.split(",") if x.strip() != ""]

    save_dir = os.path.join(os.path.dirname(args.out_summary), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    results = outer_cv_inner_holdout(
        df1_raw=df1_raw,
        df2_raw=df2_raw,
        device=device,
        n_tokens_grid=n_tokens_grid,
        d_model_grid=d_model_grid,
        nhead_grid=nhead_grid,
        num_layers_grid=num_layers_grid,
        dim_feedforward_grid=dim_feedforward_grid,
        dropout_grid=dropout_grid,
        lr_grid=lr_grid,
        bs_grid=bs_grid,
        threshold_grid=threshold_grid,
        trans1_grid=trans1_grid,
        trans2_grid=trans2_grid,
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
        save_dir=save_dir,
        eval_metric=args.eval_metric
    )

    if args.out_summary:
        out_dir = os.path.dirname(args.out_summary)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.out_summary, "w") as f:
            f.write(f"# Model: Transformer Autoencoder (V1 -> V1 denoising)\n")
            f.write(f"# Evaluation metric: {args.eval_metric}\n")
            f.write("fold\tthreshold\ttrans1\ttrans2\tn_tokens\td_model\tnhead\tnum_layers\tdim_feedforward\tdropout\tlr\tbatch_size\tzero_weight\tnonzero_weight\tinner_val_loss\tinner_val_metric\ttest_loss\ttest_metric\n")
            for i, (cfg, vl, vm, tl, tm) in enumerate(zip(
                results["best_cfgs_per_fold"],
                results["val_losses_per_fold"],
                results["val_metrics_per_fold"],
                results["test_losses_per_fold"],
                results["test_metrics_per_fold"]
            ), 1):
                f.write(f"{i}\t{cfg['threshold']}\t{cfg['trans1']}\t{cfg['trans2']}\t{cfg['n_tokens']}\t{cfg['d_model']}\t{cfg['nhead']}\t{cfg['num_layers']}\t{cfg['dim_feedforward']}\t{cfg['dropout']}\t{cfg['lr']}\t{cfg['batch_size']}\t{cfg['zero_weight']}\t{cfg['nonzero_weight']}\t{vl:.6f}\t{vm:.6f}\t{tl:.6f}\t{tm:.6f}\n")
            f.write(f"# mean_test_loss\t{results['test_loss_mean']:.6f}\n")
            f.write(f"# sd_test_loss\t{results['test_loss_sd']:.6f}\n")
            f.write(f"# mean_test_metric ({args.eval_metric})\t{results['test_metric_mean']:.6f}\n")
            f.write(f"# sd_test_metric ({args.eval_metric})\t{results['test_metric_sd']:.6f}\n")
        print(f"[SAVE] wrote summary to {args.out_summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Autoencoder: V1 -> V1 denoising")
    parser.add_argument('--data_path1', type=str, required=True, help='Path to V1 feather file (input, to denoise)')
    parser.add_argument('--data_path2', type=str, required=True, help='Path to V2 feather file (for correlation eval)')
    parser.add_argument('--threshold_grid', type=str, default="1")
    parser.add_argument('--trans1_grid', type=str, default="sqrt+1,log2,sqrt,no_trans",
                        help='Transformation types for V1')
    parser.add_argument('--trans2_grid', type=str, default="sqrt+1,log2,sqrt,no_trans",
                        help='Transformation types for V2')

    # Transformer architecture hyperparameters
    parser.add_argument('--n_tokens_grid', type=str, default="64",
                        help="Number of pseudo-tokens (e.g., 32,64,128)")
    parser.add_argument('--d_model_grid', type=str, default="128",
                        help="Transformer hidden dimension (e.g., 64,128,256)")
    parser.add_argument('--nhead_grid', type=str, default="4",
                        help="Number of attention heads (must divide d_model)")
    parser.add_argument('--num_layers_grid', type=str, default="2",
                        help="Number of Transformer encoder layers (e.g., 1,2,3)")
    parser.add_argument('--dim_feedforward_grid', type=str, default="256",
                        help="Feedforward dimension (e.g., 128,256,512)")
    parser.add_argument('--dropout_grid', type=str, default="0.0,0.1",
                        help="Dropout rate (e.g., 0.0,0.1,0.2)")

    parser.add_argument('--lr_grid', type=str, default="0.0001")
    parser.add_argument('--batch_size_grid', type=str, default="64")
    parser.add_argument('--weight_strategy', type=str, default='fixed',
                        choices=['fixed', 'sparsity_aware', 'magnitude', 'focal'])
    parser.add_argument('--zero_weight_grid', type=str, default="1.0")
    parser.add_argument('--nonzero_weight_grid', type=str, default="1.0,5.0,10.0,20.0")
    parser.add_argument('--eval_metric', type=str, default='val_loss',
                        choices=['val_loss', 'pearson', 'spearman'])
    parser.add_argument('--epochs_inner', type=int, default=60)
    parser.add_argument('--epochs_outer', type=int, default=60)
    parser.add_argument('--inner_val_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--out_summary', type=str, required=True)
    parser.add_argument('--early_stop', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--check_every', type=int, default=1)
    parser.add_argument('--outer_es_val_frac', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=5)

    args = parser.parse_args()
    main(args)