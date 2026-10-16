import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset

import pandas as pd
import argparse
import os
import re
import numpy as np
from itertools import product
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr, spearmanr
import json 
from pathlib import Path
import copy

def _check_count_matrix(df, name, check_zero=True):
    values = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name}: count matrix is empty")
    n_nan = int(np.isnan(values).sum())
    n_inf = int(np.isinf(values).sum())
    n_zero_rows = int(np.all(values == 0, axis=1).sum()) if check_zero else 0
    n_zero_cols = int(np.all(values == 0, axis=0).sum()) if check_zero else 0
    if n_nan or n_inf or n_zero_rows or n_zero_cols:
        raise ValueError(
            f"{name}: NaN={n_nan}, Inf={n_inf}, "
            f"all_zero_rows={n_zero_rows}, all_zero_cols={n_zero_cols}"
        )

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(VAE, self).__init__()
        # Encode
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
        # Decode
        self.fc3 = nn.Linear(latent_dim, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc5 = nn.Linear(hidden_dim1, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        logvar_safe = torch.clamp(logvar, min=-20.0, max=20.0)
        std = torch.exp(0.5 * logvar_safe)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return F.relu(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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
        # exact zero comparison (no transform or transform that maps 0→0)
        zero_mask = (x == 0).float()
        nonzero_mask = (x > 0).float()
    else:
        zero_mask = (torch.abs(x - zv) < tol).float()
        nonzero_mask = (x > zv + tol).float()
    return zero_mask, nonzero_mask

def weighted_reconstruction_loss(recon_x, x, weight_strategy='fixed', zero_weight=1.0, nonzero_weight=5.0, trans=None,):
    """
    Modified weighted reconstruction loss that handles different transformations.
    
    Args:
        trans: The transformation applied to the data (e.g., 'sqrt+1', 'log2', etc.)
    """
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

def vae_loss(recon_x, x, mu, logvar, beta, weight_strategy='fixed',
             zero_weight=1.0, nonzero_weight=5.0, trans=None):
    recon_loss = weighted_reconstruction_loss(
        recon_x, x, weight_strategy, zero_weight, nonzero_weight, trans
    )

    # if beta == 0, do not compute KLD at all. Otherwise 0 * inf can become NaN.
    if beta == 0:
        return recon_loss

    logvar_safe = torch.clamp(logvar, min=-20.0, max=20.0)

    n_genes = x.shape[1]
    KLD = -0.5 * torch.mean(
        torch.sum(1 + logvar_safe - mu.pow(2) - logvar_safe.exp(), dim=1)
    ) / n_genes

    return recon_loss + beta * KLD

def train_one_epoch(model, loader, optimizer, device, beta, weight_strategy='fixed',
                   zero_weight=1.0, nonzero_weight=5.0, trans=None):
    model.train()
    total = 0.0
    n = 0
    for batch_idx, (x,) in enumerate(loader):
        x = x.to(device)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"  [ERROR] NaN/Inf in input batch {batch_idx}")
            return float('inf')
        
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        
        if torch.isnan(recon).any() or torch.isinf(recon).any():
            print(f"  [ERROR] NaN/Inf in reconstruction at batch {batch_idx}")
            return float('inf')
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print(f"  [ERROR] NaN/Inf in mu at batch {batch_idx}")
            return float('inf')
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            print(f"  [ERROR] NaN/Inf in logvar at batch {batch_idx}")
            return float('inf')
        
        loss = vae_loss(recon, x, mu, logvar, beta, weight_strategy, zero_weight, nonzero_weight, trans)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [ERROR] NaN/Inf loss at batch {batch_idx}: {loss.item()}")
            print(f"    recon range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
            print(f"    x range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            return float('inf')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device, beta, weight_strategy='fixed',
              zero_weight=1.0, nonzero_weight=5.0, trans=None):
    model.eval()
    total = 0.0
    n = 0

    for batch_idx, (x,) in enumerate(loader):
        x = x.to(device)

        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"  [ERROR] NaN/Inf in eval input batch {batch_idx}, trans={trans}")
            return float("inf")

        recon, mu, logvar = model(x)

        if torch.isnan(recon).any() or torch.isinf(recon).any():
            print(f"  [ERROR] NaN/Inf in eval reconstruction at batch {batch_idx}, trans={trans}")
            return float("inf")

        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print(f"  [ERROR] NaN/Inf in eval mu at batch {batch_idx}, trans={trans}")
            return float("inf")

        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            print(f"  [ERROR] NaN/Inf in eval logvar at batch {batch_idx}, trans={trans}")
            return float("inf")

        loss = vae_loss(recon, x, mu, logvar, beta, weight_strategy, zero_weight, nonzero_weight, trans)

        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"  [ERROR] NaN/Inf eval loss at batch {batch_idx}: {loss.item()}, "
                f"trans={trans}, beta={beta}, zero_w={zero_weight}, nonzero_w={nonzero_weight}"
            )
            print(f"    recon range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
            print(f"    x range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"    mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
            print(f"    logvar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")
            return float("inf")

        total += loss.item() * x.size(0)
        n += x.size(0)

    return total / max(n, 1)


@torch.no_grad()
def eval_correlation_residual(model, loader, v2_data, indices, device, corr_type='pearson'):
    """
    Compute average residual between correlations of (v1, v2) and (vae_output, v2).
    Return corr(v1, v2) - corr(vae_output, v2).
    More negative means stronger improvement over raw v1.
    Therefore, lower is better.
    """
    model.eval()
    
    all_v1 = []
    all_recon = []
    
    for (x,) in loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        all_v1.append(x.cpu().numpy())
        all_recon.append(recon.cpu().numpy())
    
    v1_array = np.vstack(all_v1)
    vae_array = np.vstack(all_recon)
    v2_subset = v2_data[indices]
    
    n_cols = v1_array.shape[1]
    residuals = []
    
    for col_idx in range(n_cols):
        v1_col = v1_array[:, col_idx]
        vae_col = vae_array[:, col_idx]
        v2_col = v2_subset[:, col_idx]
        
        if corr_type == 'pearson':
            corr_v1_v2, _ = pearsonr(v1_col, v2_col)
        elif corr_type == 'spearman':
            corr_v1_v2, _ = spearmanr(v1_col, v2_col)
        else:
            raise ValueError(f"Unknown correlation type: {corr_type}")
        
        if np.isnan(corr_v1_v2):
            corr_v1_v2 = 0
        
        if np.any(np.isnan(vae_col)) or np.std(vae_col) == 0:
            corr_vae_v2 = 0
        else:
            if corr_type == 'pearson':
                corr_vae_v2, _ = pearsonr(vae_col, v2_col)
            elif corr_type == 'spearman':
                corr_vae_v2, _ = spearmanr(vae_col, v2_col)
            
            if np.isnan(corr_vae_v2):
                corr_vae_v2 = 0
        
        residual = corr_v1_v2 - corr_vae_v2
        residuals.append(residual)
    
    return np.mean(residuals) if residuals else 0.0

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self.stopped = False
        self.best_state = None

    def step(self, metric, model=None, epoch=None):
        if metric < self.best - self.min_delta:
            self.best = metric
            if epoch is not None:
                self.best_epoch = int(epoch)
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

def sanitize_norm_name(norm):
    """Make a normalization label safe to put in a filename without changing its meaning.

    This function MUST stay byte-identical between Step2_train_* and Step3_reconstruct_*,
    otherwise Step 2 writes a file Step 3 will not find.
    """
    return re.sub(r"[^0-9A-Za-z._+-]", "_", str(norm).strip())

def make_loader(X, idx, batch_size, shuffle):
    subset = Subset(TensorDataset(X), idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

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
    elif trans == "log2(count+2)":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 2)
    elif trans == "log2(count+1)+1":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 1) + 1
    elif trans == "no_trans":
        pass
    else:
        raise ValueError(f"Unknown transformation: {trans}")


def filter_and_transform(df1, df2, threshold_value, trans1, trans2, data_path1=None, data_path2=None, save=False):
    _check_count_matrix(df1, "V1 before filtering")
    _check_count_matrix(df2, "V2 before filtering", check_zero=False)
    data_cols = df1.columns[1:]
    zero_percentage = (df1[data_cols] == 0).mean()
    keep_cols = zero_percentage < threshold_value

    pos_col = df1.columns[0]
    cols_to_keep = [pos_col] + data_cols[keep_cols].tolist()

    filtered_df1 = df1[cols_to_keep].copy()
    filtered_df2 = df2[cols_to_keep].copy()
    _check_count_matrix(filtered_df1, "V1 before transformation")
    _check_count_matrix(filtered_df2, "V2 before transformation", check_zero=False)

    _apply_trans(filtered_df1, trans1)
    _apply_trans(filtered_df2, trans2)
    _check_count_matrix(filtered_df1, "V1 after transformation", check_zero=False)
    _check_count_matrix(filtered_df2, "V2 after transformation", check_zero=False)

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
    
    # Extra safety check: ensure all columns are numeric
    for df, name in [(df1, 'df1'), (df2, 'df2')]:
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"[WARNING] Converting object column {col} in {name}")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name}: NaN/Inf present before tensor conversion")
        
        if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError(f"Cannot convert to tensor: non-numeric data present in {name}")
    
    X = torch.tensor(df1.to_numpy(), dtype=torch.float32)
    if not torch.isfinite(X).all():
        raise ValueError("V1: NaN/Inf present after float32 conversion")
    X2 = df2.to_numpy()
    return X, X2


def outer10_inner_holdout(
    df1_raw, df2_raw, device,
    hidden_grid1, hidden_grid2, latent_grid, lr_grid, bs_grid, beta_grid,
    threshold_grid, trans1_grid, trans2_grid,
    epochs_inner, epochs_outer, inner_val_frac, seed,
    early_stop=False, patience=10, min_delta=0.0, check_every=1, outer_es_val_frac=0.1,
    n_splits=10, weight_strategy='fixed', zero_weight_grid=[1.0], nonzero_weight_grid=[5.0],
    save_dir=None, eval_metric='val_loss', normalization=None):

    if save_dir is None:
        raise ValueError("save_dir must be provided")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("[DATA VALIDATION]")
    print("="*60)
    print(f"  df1_raw shape: {df1_raw.shape}")
    print(f"  df2_raw shape: {df2_raw.shape}")
    
    df1_numeric = df1_raw.drop(columns=['pos'], errors='ignore').select_dtypes(include=[np.number])
    df2_numeric = df2_raw.drop(columns=['pos'], errors='ignore').select_dtypes(include=[np.number])
    
    df1_nan_count = df1_numeric.isna().sum().sum()
    df1_inf_count = np.isinf(df1_numeric).sum().sum()
    df1_min = df1_numeric.min().min()
    df1_max = df1_numeric.max().max()
    df1_mean = df1_numeric.mean().mean()
    
    print(f"\n  df1_raw statistics:")
    print(f"    NaN count: {df1_nan_count}")
    print(f"    Inf count: {df1_inf_count}")
    print(f"    Range: [{df1_min:.4f}, {df1_max:.4f}]")
    print(f"    Mean: {df1_mean:.4f}")
    
    df2_nan_count = df2_numeric.isna().sum().sum()
    df2_inf_count = np.isinf(df2_numeric).sum().sum()
    df2_min = df2_numeric.min().min()
    df2_max = df2_numeric.max().max()
    df2_mean = df2_numeric.mean().mean()
    
    print(f"\n  df2_raw statistics:")
    print(f"    NaN count: {df2_nan_count}")
    print(f"    Inf count: {df2_inf_count}")
    print(f"    Range: [{df2_min:.4f}, {df2_max:.4f}]")
    print(f"    Mean: {df2_mean:.4f}")
    
    df1_variance = df1_numeric.var().mean()
    df2_variance = df2_numeric.var().mean()
    print(f"\n  df1_raw mean variance: {df1_variance:.6f}")
    print(f"  df2_raw mean variance: {df2_variance:.6f}")
    
    if df1_nan_count > 0 or df2_nan_count > 0:
        print("\n  [WARNING] NaN values found in input data!")
    if df1_inf_count > 0 or df2_inf_count > 0:
        print("  [WARNING] Inf values found in input data!")
    if df1_variance == 0 or df2_variance == 0:
        print("  [WARNING] No variance in data!")
    if df1_max > 1e6 or df2_max > 1e6:
        print("  [WARNING] Very large values detected! Consider normalization.")
    
    print("="*60 + "\n")
    
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

    # --- transformation and normalization are EXPERIMENTAL CONDITIONS, not model
    #     hyperparameters: one (transformation, normalization) combination is trained
    #     per call of this script, so they are excluded from the configuration count. ---
    if len(threshold_grid) != 1 or len(trans1_grid) != 1 or len(trans2_grid) != 1:
        raise ValueError(
            "threshold / trans1 / trans2 are experimental conditions, not hyperparameters. "
            "This script trains exactly one (transformation, normalization) combination per "
            f"call, but received {len(threshold_grid)} threshold, {len(trans1_grid)} trans1 "
            f"and {len(trans2_grid)} trans2 value(s). Loop over them in the driver.")
    if normalization is None:
        raise ValueError("normalization must be provided (it names the final model file)")
    norm_tag = sanitize_norm_name(normalization)

    threshold, trans1, trans2 = threshold_grid[0], trans1_grid[0], trans2_grid[0]
    X, X2 = tensor_cache[(threshold, trans1, trans2)]
    input_dim = X.shape[1]
    n_cells = int(X.shape[0])
    all_cell_idx = np.arange(n_cells)

    MODEL_PARAM_NAMES = ["hidden_dim1", "hidden_dim2", "latent_dim", "lr", "batch_size", "beta", "zero_weight", "nonzero_weight"]
    model_param_grids = [hidden_grid1, hidden_grid2, latent_grid, lr_grid, bs_grid, beta_grid,
                         zero_weight_grid, nonzero_weight_grid]
    model_configs = [dict(zip(MODEL_PARAM_NAMES, combo))
                     for combo in product(*model_param_grids)]
    needs_selection = len(model_configs) > 1

    print("\n" + "=" * 60)
    print(f"[PLAN] transformation={trans1}  normalization={normalization}  "
          f"({n_cells} cells, {input_dim} genes)")
    print(f"[PLAN] {len(model_configs)} model configuration(s) -> "
          + ("nested cross-validation, then a refit on all cells"
             if needs_selection else
             "no selection needed; best_epoch pass, then a refit on all cells"))
    print("=" * 60, flush=True)

    def _fit(cfg, train_idx, max_epochs, val_idx=None):
        """Train one configuration from a fresh, identically seeded initialisation.

        val_idx given -> early-stop on it, restore the best checkpoint, and return the
                         epoch at which it was reached.
        val_idx None  -> train exactly max_epochs with no validation and no early
                         stopping (used for every model that is actually kept).
        Returns (model, best_epoch, best_metric, val_loss_at_best).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = VAE(input_dim, cfg["hidden_dim1"], cfg["hidden_dim2"],
                    cfg["latent_dim"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
        train_loader = make_loader(X, train_idx, batch_size=cfg["batch_size"], shuffle=True)

        if val_idx is None:
            for ep in range(1, int(max_epochs) + 1):
                train_one_epoch(model, train_loader, optimizer, device, cfg["beta"],
                                weight_strategy, cfg["zero_weight"],
                                cfg["nonzero_weight"], trans1)
            return model, int(max_epochs), None, None

        val_loader = make_loader(X, val_idx, batch_size=cfg["batch_size"], shuffle=False)
        es = EarlyStopping(patience=patience, min_delta=min_delta)
        for ep in range(1, int(max_epochs) + 1):
            train_one_epoch(model, train_loader, optimizer, device, cfg["beta"],
                                weight_strategy, cfg["zero_weight"],
                                cfg["nonzero_weight"], trans1)
            if ep % check_every == 0:
                if eval_metric == 'val_loss':
                    curr = eval_loss(model, val_loader, device, cfg["beta"], weight_strategy,
                                     cfg["zero_weight"], cfg["nonzero_weight"], trans1)
                elif eval_metric in ['pearson', 'spearman']:
                    curr = eval_correlation_residual(model, val_loader, X2, val_idx,
                                                     device, eval_metric)
                else:
                    raise ValueError(f"Unknown eval_metric: {eval_metric}")
                if es.step(curr, model, epoch=ep):
                    break
        # restore the best checkpoint whether or not early stopping fired
        if es.best_state is not None:
            model.load_state_dict(es.best_state)
        v_loss = eval_loss(model, val_loader, device, cfg["beta"], weight_strategy,
                                     cfg["zero_weight"], cfg["nonzero_weight"], trans1)
        return model, int(max(1, es.best_epoch)), float(es.best), float(v_loss)

    def _evaluate(model, idx, cfg):
        loader = make_loader(X, idx, batch_size=cfg["batch_size"], shuffle=False)
        val_loader = loader
        loss = eval_loss(model, val_loader, device, cfg["beta"], weight_strategy,
                                     cfg["zero_weight"], cfg["nonzero_weight"], trans1)
        if eval_metric == 'val_loss':
            metric = loss
        else:
            metric = eval_correlation_residual(model, loader, X2, idx, device, eval_metric)
        return float(loss), float(metric)

    fold_best_cfgs, fold_val_losses, fold_val_metrics = [], [], []
    fold_test_losses, fold_test_metrics, fold_best_epochs = [], [], []
    all_val_results = []
    inner_metric_by_config, inner_epoch_by_config = {}, {}

    if not needs_selection:
        # ---------- single configuration: two passes, no KFold, no outer test ----------
        final_cfg = model_configs[0]
        tr_idx, va_idx = train_test_split(all_cell_idx, test_size=inner_val_frac,
                                          random_state=seed, shuffle=True)
        print(f"\n[pass 1] {len(tr_idx)} train / {len(va_idx)} validation cells, "
              f"early stopping on validation to pick best_epoch", flush=True)
        tmp_model, final_best_epoch, final_inner_metric, final_inner_loss = _fit(
            final_cfg, tr_idx, epochs_inner, val_idx=va_idx)
        del tmp_model          # the pass-1 model only supplies best_epoch; it is discarded
        print(f"[pass 1] best_epoch={final_best_epoch}  "
              f"val_metric({eval_metric})={final_inner_metric:.6f}", flush=True)

        all_val_results.append(dict(config_name=str(final_cfg), fold=0,
                                    val_metric=float(final_inner_metric),
                                    val_loss=float(final_inner_loss),
                                    best_epoch=int(final_best_epoch)))
        selection_mode, n_splits_used = "single_config", 0
        outer_test_loss_mean = outer_test_loss_sd = None
        outer_test_metric_mean = outer_test_metric_sd = None
    else:
        # ---------------------- nested cross-validation ----------------------
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_id, (outer_train_idx, outer_test_idx) in enumerate(kf.split(df1_raw), 1):
            print(f"\n========== Outer fold {fold_id}/{n_splits} ==========", flush=True)
            tr_idx, val_idx = train_test_split(outer_train_idx, test_size=inner_val_frac,
                                               random_state=seed, shuffle=True)
            best_ci, best_val, best_val_loss, best_ep = None, float("inf"), float("inf"), None

            for ci, cfg in enumerate(model_configs):
                tmp_model, ep_i, m_i, l_i = _fit(cfg, tr_idx, epochs_inner, val_idx=val_idx)
                del tmp_model      # inner models are temporary: they only rank configs
                if m_i is None or not np.isfinite(m_i):
                    print(f"  [WARNING] fold {fold_id} config {cfg} gave "
                          f"val_metric={m_i}; skipped", flush=True)
                    continue
                inner_metric_by_config.setdefault(ci, []).append(float(m_i))
                inner_epoch_by_config.setdefault(ci, []).append(int(ep_i))
                all_val_results.append(dict(config_name=str(cfg), fold=fold_id,
                                            val_metric=float(m_i), val_loss=float(l_i),
                                            best_epoch=int(ep_i)))
                if m_i < best_val:
                    best_ci, best_val, best_val_loss, best_ep = ci, float(m_i), float(l_i), int(ep_i)

            if best_ci is None:
                raise ValueError(f"[Fold {fold_id}] no configuration produced a finite "
                                 f"inner validation metric")
            fold_cfg = model_configs[best_ci]
            print(f"[Fold {fold_id}] inner-selected config: {fold_cfg}  "
                  f"inner_val_metric={best_val:.6f}  best_epoch={best_ep}", flush=True)

            # refit the fold's config on the FULL outer training set for its best_epoch,
            # with no validation split and no early stopping
            fold_model, _, _, _ = _fit(fold_cfg, outer_train_idx, best_ep, val_idx=None)

            # the outer test set is touched exactly once, here, and only to evaluate
            t_loss, t_metric = _evaluate(fold_model, outer_test_idx, fold_cfg)
            print(f"[Fold {fold_id}] outer test (evaluation only): "
                  f"loss={t_loss:.6f}  metric({eval_metric})={t_metric:.6f}", flush=True)

            # the summary writer also prints the experimental conditions, so they are
            # merged into the per-fold record; model_configs itself stays pure
            fold_best_cfgs.append(dict(threshold=threshold, trans1=trans1,
                                       trans2=trans2, **fold_cfg))
            fold_val_losses.append(best_val_loss)
            fold_val_metrics.append(best_val)
            fold_test_losses.append(t_loss)
            fold_test_metrics.append(t_metric)
            fold_best_epochs.append(int(best_ep))

            fold_dir = os.path.join(save_dir, f"fold_{fold_id}")
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(fold_model.state_dict(), os.path.join(fold_dir, "weights.pt"))
            with open(os.path.join(fold_dir, "metrics.json"), "w") as fh:
                json.dump(dict(fold=fold_id, config=fold_cfg, best_epoch=int(best_ep),
                               inner_val_metric=best_val, inner_val_loss=best_val_loss,
                               outer_test_loss=t_loss, outer_test_metric=t_metric,
                               note=("outer test metrics are for evaluation only and are "
                                     "never used to select a configuration, an epoch, a "
                                     "fold or the final model")), fh, indent=2)
            del fold_model

        # final configuration comes from the INNER validation results only
        mean_inner = {ci: float(np.mean(v)) for ci, v in inner_metric_by_config.items()
                      if len(v) > 0}
        if not mean_inner:
            raise ValueError("no configuration produced a finite inner validation metric")
        best_ci = min(mean_inner, key=lambda c: mean_inner[c])
        final_cfg = model_configs[best_ci]
        final_inner_metric = mean_inner[best_ci]
        final_best_epoch = int(max(1, round(float(np.median(inner_epoch_by_config[best_ci])))))
        selection_mode, n_splits_used = "nested_cv", int(n_splits)

        outer_test_loss_mean = float(np.mean(fold_test_losses))
        outer_test_loss_sd = float(np.std(fold_test_losses, ddof=1)) if len(fold_test_losses) > 1 else 0.0
        outer_test_metric_mean = float(np.mean(fold_test_metrics))
        outer_test_metric_sd = float(np.std(fold_test_metrics, ddof=1)) if len(fold_test_metrics) > 1 else 0.0

        print(f"\n[OUTER TEST, evaluation only] loss  mean={outer_test_loss_mean:.6f} "
              f"sd={outer_test_loss_sd:.6f}")
        print(f"[OUTER TEST, evaluation only] {eval_metric} mean={outer_test_metric_mean:.6f} "
              f"sd={outer_test_metric_sd:.6f}")
        print(f"[SELECT] final config from mean inner validation metric: {final_cfg}")
        print(f"[SELECT] final best_epoch = median of {inner_epoch_by_config[best_ci]} "
              f"= {final_best_epoch}", flush=True)

    # ---------------- the one model that is kept: all cells, fixed epochs ----------------
    print(f"\n========== Final model: all {n_cells} cells, {final_best_epoch} epochs, "
          f"no validation split ==========", flush=True)
    final_model, _, _, _ = _fit(final_cfg, all_cell_idx, final_best_epoch, val_idx=None)

    final_weights_path = os.path.join(save_dir, f"final_model_{norm_tag}.pt")
    final_config_path = os.path.join(save_dir, f"final_config_{norm_tag}.json")
    torch.save(final_model.state_dict(), final_weights_path)
    with open(final_config_path, "w") as fh:
        json.dump({
            "normalization": str(normalization),
            "transformation": str(trans1),
            "trans1": str(trans1),
            "trans2": str(trans2),
            "threshold": float(threshold),
            "input_dim": int(input_dim),
            "hidden_dim1": final_cfg["hidden_dim1"],
            "hidden_dim2": final_cfg["hidden_dim2"],
            "latent_dim": final_cfg["latent_dim"],
            "lr": final_cfg["lr"],
            "batch_size": final_cfg["batch_size"],
            "beta": final_cfg["beta"],
            "zero_weight": final_cfg["zero_weight"],
            "nonzero_weight": final_cfg["nonzero_weight"],
            "weight_strategy": weight_strategy,
            "eval_metric": eval_metric,
            "final_best_epoch": int(final_best_epoch),
            "selection_mode": selection_mode,
            "n_splits_used": int(n_splits_used),
            "inner_validation_metric": float(final_inner_metric),
            "outer_test_loss_mean": outer_test_loss_mean,
            "outer_test_loss_sd": outer_test_loss_sd,
            "outer_test_metric_mean": outer_test_metric_mean,
            "outer_test_metric_sd": outer_test_metric_sd,
            "n_cells_total": int(n_cells),
            "n_cells_trained_on": int(n_cells),
            "seed": int(seed)
        }, fh, indent=2)
    print(f"[SAVE] final model  -> {final_weights_path}")
    print(f"[SAVE] final config -> {final_config_path}", flush=True)

    val_df = pd.DataFrame(all_val_results)
    if val_df.empty:
        val_df = pd.DataFrame(columns=["config_name", "fold", "val_metric",
                                       "val_loss", "best_epoch"])
        val_df_grouped = val_df.copy()
    else:
        val_df_grouped = (val_df.groupby("config_name", as_index=False)
                          .agg(mean_val_metric=("val_metric", "mean"),
                               sd_val_metric=("val_metric", "std"),
                               median_best_epoch=("best_epoch", "median"),
                               n_folds=("fold", "nunique"))
                          .sort_values("mean_val_metric"))
    val_path = os.path.join(save_dir, f"all_validation_results_{norm_tag}.csv")
    val_df_grouped.to_csv(val_path, index=False)
    print(f"[SAVE] inner validation results -> {val_path}")

    return {
        "best_cfgs_per_fold": fold_best_cfgs,
        "val_losses_per_fold": fold_val_losses,
        "val_metrics_per_fold": fold_val_metrics,
        "test_losses_per_fold": fold_test_losses,
        "test_metrics_per_fold": fold_test_metrics,
        "best_epochs_per_fold": fold_best_epochs,
        "test_loss_mean": outer_test_loss_mean,
        "test_loss_sd": outer_test_loss_sd,
        "test_metric_mean": outer_test_metric_mean,
        "test_metric_sd": outer_test_metric_sd,
        "eval_metric": eval_metric,
        "selection_mode": selection_mode,
        "n_splits_used": int(n_splits_used),
        "final_config": final_cfg,
        "final_best_epoch": int(final_best_epoch),
        "inner_validation_metric": float(final_inner_metric),
        "normalization": str(normalization),
        "norm_tag": norm_tag,
        "final_weights_path": final_weights_path,
        "final_config_path": final_config_path,
        "n_cells_total": int(n_cells),
        "validation_df": val_df_grouped,
    }

def _fmt(v):
    return "null" if v is None else f"{v:.6f}"


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load raw data
    df1_raw = pd.read_feather(args.data_path1)
    df2_raw = pd.read_feather(args.data_path2)

    hidden_grid1 = parse_grid(args.hidden_grid1, int)
    hidden_grid2 = parse_grid(args.hidden_grid2, int)
    latent_grid = parse_grid(args.latent_grid, int)
    lr_grid = [float(x) for x in args.lr_grid.split(",") if x.strip() != ""]
    bs_grid = parse_grid(args.batch_size_grid, int)
    beta_grid = [float(x) for x in args.beta_grid.split(",") if x.strip() != ""]
    zero_weight_grid = [float(x) for x in args.zero_weight_grid.split(",") if x.strip() != ""]
    nonzero_weight_grid = [float(x) for x in args.nonzero_weight_grid.split(",") if x.strip() != ""]
    
    threshold_grid = [float(x) for x in args.threshold_grid.split(",") if x.strip() != ""]
    trans1_grid = [x.strip() for x in args.trans1_grid.split(",") if x.strip() != ""]
    trans2_grid = [x.strip() for x in args.trans2_grid.split(",") if x.strip() != ""]

    save_dir = os.path.join(os.path.dirname(args.out_summary), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    results = outer10_inner_holdout(
        df1_raw=df1_raw,
        df2_raw=df2_raw,
        device=device,
        hidden_grid1=hidden_grid1,
        hidden_grid2=hidden_grid2,
        latent_grid=latent_grid,
        lr_grid=lr_grid,
        bs_grid=bs_grid,
        beta_grid=beta_grid,
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
        normalization=args.normalization,
        eval_metric=args.eval_metric
    )

    if args.out_summary:
        out_dir = os.path.dirname(args.out_summary)
        if out_dir: 
            os.makedirs(out_dir, exist_ok=True)

        with open(args.out_summary, "w") as f:
            f.write(f"# Evaluation metric: {args.eval_metric}\n")
            f.write("fold\tthreshold\ttrans1\ttrans2\thidden_dim1\thidden_dim2\tlatent_dim\tlr\tbatch_size\tbeta\tzero_weight\tnonzero_weight\tinner_val_loss\tinner_val_metric\ttest_loss\ttest_metric\n")
            for i, (cfg, vl, vm, tl, tm) in enumerate(zip(
                results["best_cfgs_per_fold"],
                results["val_losses_per_fold"],
                results["val_metrics_per_fold"],
                results["test_losses_per_fold"],
                results["test_metrics_per_fold"]
            ), 1):
                f.write(f"{i}\t{cfg['threshold']}\t{cfg['trans1']}\t{cfg['trans2']}\t{cfg['hidden_dim1']}\t{cfg['hidden_dim2']}\t{cfg['latent_dim']}\t{cfg['lr']}\t{cfg['batch_size']}\t{cfg['beta']}\t{cfg['zero_weight']}\t{cfg['nonzero_weight']}\t{vl:.6f}\t{vm:.6f}\t{tl:.6f}\t{tm:.6f}\n")
            f.write(f"# mean_test_loss\t{_fmt(results['test_loss_mean'])}\n")
            f.write(f"# sd_test_loss\t{_fmt(results['test_loss_sd'])}\n")
            f.write(f"# mean_test_metric ({args.eval_metric})\t{_fmt(results['test_metric_mean'])}\n")
            f.write(f"# sd_test_metric ({args.eval_metric})\t{_fmt(results['test_metric_sd'])}\n")
        print(f"[SAVE] wrote summary to {args.out_summary}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE with Weighted Reconstruction Loss and Multiple Evaluation Metrics")
    parser.add_argument('--data_path1', type=str, required=True, help='Path to first feather file (rep1)')
    parser.add_argument('--data_path2', type=str, required=True, help='Path to second feather file (rep2)')
    parser.add_argument('--threshold_grid', type=str, default="1", help='Threshold values for filtering')
    parser.add_argument('--trans1_grid', type=str, default="sqrt+1,log2,sqrt,no_trans", help='Transformation types for V1')
    parser.add_argument('--trans2_grid', type=str, default="no_trans", help='Transformation types for V2')
    parser.add_argument('--hidden_grid1', type=str, default="512,1024,2048,4096")
    parser.add_argument('--hidden_grid2', type=str, default="128,256,512,1024")
    parser.add_argument('--latent_grid', type=str, default="32")
    parser.add_argument('--lr_grid', type=str, default="0.0001")
    parser.add_argument('--batch_size_grid', type=str, default="64")
    parser.add_argument('--beta_grid', type=str, default="0")
    parser.add_argument('--weight_strategy', type=str, default='fixed', 
                       choices=['fixed', 'sparsity_aware', 'magnitude', 'focal'])
    parser.add_argument('--zero_weight_grid', type=str, default="1.0")
    parser.add_argument('--nonzero_weight_grid', type=str, default="1.0")
    parser.add_argument('--eval_metric', type=str, default='val_loss', 
                       choices=['val_loss', 'pearson', 'spearman'],
                       help='Evaluation metric for hyperparameter selection')
    parser.add_argument('--epochs_inner', type=int, default=60)
    parser.add_argument('--epochs_outer', type=int, default=60)
    parser.add_argument('--inner_val_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--normalization', type=str, required=True,
                        help='normalization label from the driver; names the final model file')
    parser.add_argument('--out_summary', type=str, required=True)
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--check_every', type=int, default=1)
    parser.add_argument('--outer_es_val_frac', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=5)

    args = parser.parse_args()
    main(args)
