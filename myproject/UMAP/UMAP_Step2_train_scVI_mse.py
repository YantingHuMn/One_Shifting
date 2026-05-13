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
from pathlib import Path
import collections


#  FCLayers – faithful re-implementation of scvi.nn.FCLayers
class FCLayers(nn.Module):
    """Fully-connected layers with optional BatchNorm, LayerNorm, covariate injection.

    Network architecture faithfully reproduces scvi-tools VAE:
    Encoder:  FCLayers (with BatchNorm, inject_covariates) → mu / logvar → reparameterize
    Library:  FCLayers (1-layer) → log-library mean / var  (observed library size shortcut)
    Decoder:  FCLayers (with BatchNorm, inject_covariates) → Softmax(scale) * library → rate
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: list[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn=nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates

        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.n_cov = sum(self.n_cat_list)

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                _n_in + self.n_cov * self._inject_into_layer(i),
                                _n_out,
                                bias=bias,
                            ),
                            nn.BatchNorm1d(_n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(_n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (_n_in, _n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def _inject_into_layer(self, layer_num) -> bool:
        return layer_num == 0 or (layer_num > 0 and self.inject_covariates)

    def forward(self, x: torch.Tensor, *cat_list):
        one_hot_cat_list = []
        cat_list = cat_list or []
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0")
            if n_cat > 1:
                if cat.dim() == 1:
                    cat = cat.unsqueeze(-1)
                if cat.size(-1) != n_cat:
                    one_hot_cat = F.one_hot(cat.squeeze(-1), n_cat).float()
                else:
                    one_hot_cat = cat.float()
                one_hot_cat_list.append(one_hot_cat)

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self._inject_into_layer(i):
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        x = layer(x)
        return x


#  Encoder – scVI style (returns Normal dist params + reparameterised sample)
class EncoderSCVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: list[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        inject_covariates: bool = True,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=inject_covariates,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list):
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        log_q_v = torch.clamp(self.var_encoder(q), min=-20.0, max=20.0)
        q_v = torch.exp(log_q_v) + self.var_eps
        dist = torch.distributions.Normal(q_m, q_v.sqrt())
        z = dist.rsample()
        return q_m, q_v, z


#  Decoder – scVI style (FCLayers → linear heads for scale, rate, dropout)
class DecoderSCVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: list[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor, library: torch.Tensor, *cat_list):
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_rate = torch.exp(library) * px_scale
        return px_rate


#  Full scVI-structure model  (encoder + library + decoder, no distribution)
class ScVIModel(nn.Module):
    """
    scVI architecture (encoder → z, decoder → reconstruction)
    with plain MSE loss instead of negative-binomial likelihood.

    Parameters mirror scVI defaults:
      n_layers=1, n_hidden=128, n_latent=10, dropout_rate=0.1,
      use_batch_norm encoder+decoder, log_variational=False,
      use_observed_lib_size=True
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_batch_norm_encoder: bool = True,
        use_batch_norm_decoder: bool = True,
        use_layer_norm_encoder: bool = False,
        use_layer_norm_decoder: bool = False,
        log_variational: bool = False,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.log_variational = log_variational

        # z encoder: data → latent
        self.z_encoder = EncoderSCVI(
            n_input=n_input,
            n_output=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        # decoder: latent → gene rate
        self.decoder = DecoderSCVI(
            n_input=n_latent,
            n_output=n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        recon : reconstructed rate (n_obs, n_genes)
        mu    : latent mean
        logvar: latent log-variance (log of q_v)
        """
        # encode
        if self.log_variational:
            library = torch.log(x.sum(dim=1, keepdim=True) + 1e-6)
            x_input = torch.log1p(x)
        else:
            library = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
            x_input = x

        q_m, q_v, z = self.z_encoder(x_input)

        # decode
        px_rate = self.decoder(z, library)

        # logvar for KL computation: q_v = exp(logvar) + eps  →  logvar ≈ log(q_v)
        logvar = torch.log(q_v)

        return px_rate, q_m, logvar


#  Weighted MSE loss  (identical logic to DCA / VAE scripts)
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


def vae_loss(recon_x, x, mu, logvar, beta,
             weight_strategy='fixed', zero_weight=1.0, nonzero_weight=5.0, trans=None):
    recon_loss = weighted_reconstruction_loss(
        recon_x, x, weight_strategy, zero_weight, nonzero_weight, trans
    )

    if beta == 0:
        return recon_loss

    logvar_safe = torch.clamp(logvar, min=-20.0, max=20.0)

    KLD = -0.5 * torch.mean(
        torch.sum(1 + logvar_safe - mu.pow(2) - logvar_safe.exp(), dim=1)
    )

    return recon_loss + beta * KLD


#  Train / eval helpers
def train_one_epoch(model, loader, optimizer, device, beta, weight_strategy='fixed', zero_weight=1.0, nonzero_weight=5.0, trans=None,):
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

        loss = vae_loss(recon, x, mu, logvar, beta, weight_strategy, zero_weight, nonzero_weight, trans)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [ERROR] NaN/Inf loss at batch {batch_idx}: {loss.item()}")
            return float('inf')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device, beta, weight_strategy='fixed', zero_weight=1.0, nonzero_weight=5.0, trans=None,):
    model.eval()
    total = 0.0
    n = 0
    for (x,) in loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar, beta, weight_strategy, zero_weight, nonzero_weight, trans)
        total += loss.item()
        n += x.size(0)
    return total / max(n, 1)


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

def filter_and_transform(df1, threshold_value, trans1,
                         data_path1=None, save=False):
    data_cols = df1.columns[1:]
    zero_percentage = (df1[data_cols] == 0).mean()
    keep_cols = zero_percentage < threshold_value

    pos_col = df1.columns[0]
    cols_to_keep = [pos_col] + data_cols[keep_cols].tolist()

    filtered_df1 = df1[cols_to_keep].copy()

    _apply_trans(filtered_df1, trans1)

    for col in filtered_df1.columns[1:]:
        if filtered_df1[col].dtype == 'object':
            filtered_df1[col] = pd.to_numeric(filtered_df1[col], errors='coerce')

    filtered_df1.iloc[:, 1:] = filtered_df1.iloc[:, 1:].fillna(0)

    if save:
        if data_path1 is not None:
            save_dir1 = Path(data_path1).parent
            save_dir1.mkdir(parents=True, exist_ok=True)
            filtered_df1.to_feather(save_dir1 / f"Count_matrix_transformed_{trans1}.feather")

    return filtered_df1


def _prepare_tensors(filtered_df1):
    df1 = filtered_df1.copy()
    if 'pos' in df1.columns:
        df1.drop(columns=['pos'], inplace=True)
    elif 'barcode' in df1.columns:
        df1.drop(columns=['barcode'], inplace=True)
    for col in df1.columns:
        if df1[col].dtype == 'object':
            df1[col] = pd.to_numeric(df1[col], errors='coerce')
    df1.fillna(0, inplace=True)
    X = torch.tensor(df1.to_numpy(), dtype=torch.float32)
    return X


#  Outer K-fold + inner hold-out  (same structure as your VAE script)
def outer10_inner_holdout(
    df1_raw, device,
    hidden_grid, latent_grid, n_layers_grid, lr_grid, bs_grid, beta_grid,
    threshold_grid, trans1_grid,
    epochs_inner, epochs_outer, inner_val_frac, seed,
    early_stop=False, patience=10, min_delta=0.0, check_every=1, outer_es_val_frac=0.1,
    n_splits=10, weight_strategy='fixed',
    zero_weight_grid=None, nonzero_weight_grid=None,
    save_dir=None, eval_metric='val_loss',):

    if zero_weight_grid is None:
        zero_weight_grid = [1.0]
    if nonzero_weight_grid is None:
        nonzero_weight_grid = [5.0]
    if save_dir is None:
        raise ValueError("save_dir must be provided")
    os.makedirs(save_dir, exist_ok=True)

    # eval_metric must be val_loss since there is no second dataset
    if eval_metric != 'val_loss':
        print(f"[WARNING] eval_metric='{eval_metric}' requires a second dataset. "
              f"Falling back to 'val_loss'.")
        eval_metric = 'val_loss'

    # data validation
    print("\n" + "=" * 60)
    print("[DATA VALIDATION]")
    print("=" * 60)
    print(f"  df1_raw shape: {df1_raw.shape}")

    df1_numeric = df1_raw.drop(columns=['pos'], errors='ignore').select_dtypes(include=[np.number])
    print(f"  df1 range: [{df1_numeric.min().min():.4f}, {df1_numeric.max().max():.4f}]")
    print("=" * 60 + "\n")

    # --- Negative-value safety check ---
    df1_numeric = df1_raw.iloc[:, 1:]
    has_negatives_df1 = (df1_numeric < 0).any().any()
    if has_negatives_df1:
        unsafe = {'sqrt', 'sqrt+1', 'sqrt+0.00001', 'sqrt+10', 'sqrt+1_then_minus_1',
                  'log2', 'log2_then_add_1', 'log2(count+2)', 'log2(count+1)+1', 'log(count+2)'}
        trans1_grid = [t for t in trans1_grid if t not in unsafe] or ['no_trans']
        print(f"[INFO] Negative values detected in df1, "
              f"filtered trans1 grid to: {trans1_grid}")

    # pre-compute transforms
    transform_cache = {}
    tensor_cache = {}
    for threshold, trans1 in product(threshold_grid, trans1_grid):
        key = (threshold, trans1)
        if key not in transform_cache:
            fdf1 = filter_and_transform(df1_raw, threshold, trans1)
            transform_cache[key] = fdf1
            tensor_cache[key] = _prepare_tensors(fdf1)
    print(f"[CACHE] Pre-computed {len(transform_cache)} (threshold, trans1) combinations.\n")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_best_cfgs = []
    fold_val_losses = []
    fold_test_losses = []
    fold_val_metrics = []
    fold_test_metrics = []
    all_val_results = []

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(kf.split(df1_raw), 1):
        print(f"\n========== Fold {fold_id}/{n_splits} ==========")
        tr_idx, val_idx = train_test_split(
            outer_train_idx, test_size=inner_val_frac, random_state=seed, shuffle=True
        )

        best_cfg = None
        best_val = float("inf")
        best_val_loss = float("inf")
        fold_val_combinations = []

        print(f"[Fold {fold_id}] Evaluating all hyperparameter combinations on validation set...")
        for (threshold, trans1, n_hidden, n_latent, n_layers,
             lr, bs, beta, zero_w, nonzero_w) in product(
                threshold_grid, trans1_grid,
                hidden_grid, latent_grid, n_layers_grid,
                lr_grid, bs_grid, beta_grid,
                zero_weight_grid, nonzero_weight_grid):

            cache_key = (threshold, trans1)
            X = tensor_cache[cache_key]
            input_dim = X.shape[1]

            tr_loader = make_loader(X, tr_idx, batch_size=bs, shuffle=True)
            val_loader = make_loader(X, val_idx, batch_size=bs, shuffle=False)

            model = ScVIModel(
                n_input=input_dim,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                dropout_rate=0.1,
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            if early_stop:
                es_inner = EarlyStopping(patience=patience, min_delta=min_delta)
            for ep in range(1, epochs_inner + 1):
                train_one_epoch(model, tr_loader, optimizer, device, beta,
                                weight_strategy, zero_w, nonzero_w, trans1)
                if early_stop and (ep % check_every == 0):
                    curr_val = eval_loss(model, val_loader, device, beta,
                                         weight_strategy, zero_w, nonzero_w, trans1)
                    if es_inner.step(curr_val, model):
                        print(f"    [inner] early-stopped at epoch {ep}, "
                              f"best_val={es_inner.best:.4f}", flush=True)
                        if es_inner.best_state is not None:
                            model.load_state_dict(es_inner.best_state)
                        break

            val_metric = eval_loss(model, val_loader, device, beta,
                                   weight_strategy, zero_w, nonzero_w, trans1)
            val_loss = val_metric

            config_name = (f"{zero_w}_{nonzero_w}_{trans1}_{threshold}_"
                           f"{n_hidden}_{n_latent}_{n_layers}_{beta}")
            fold_val_combinations.append({
                'config_name': config_name, 'val_metric': val_metric, 'fold': fold_id
            })

            if val_metric < best_val:
                best_val = val_metric
                best_val_loss = val_loss
                best_cfg = dict(
                    threshold=threshold, trans1=trans1,
                    n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                    lr=lr, batch_size=bs, beta=beta,
                    zero_weight=zero_w, nonzero_weight=nonzero_w,
                )

        all_val_results.extend(fold_val_combinations)
        if best_cfg is None:
            raise ValueError(f"[Fold {fold_id}] No valid config found.")

        print(f"[Fold {fold_id}] Best config: {best_cfg}, "
              f"val_metric({eval_metric})={best_val:.4f}")
        fold_best_cfgs.append(best_cfg)
        fold_val_losses.append(best_val_loss)
        fold_val_metrics.append(best_val)

        # --- retrain best config on full train, evaluate on test ---
        print(f"[Fold {fold_id}] Retraining best config on full training set...")
        c = best_cfg
        cache_key = (c["threshold"], c["trans1"])
        X = tensor_cache[cache_key]
        input_dim = X.shape[1]

        use_outer_val = outer_es_val_frac > 0.0
        if use_outer_val:
            tr_full_idx, outer_val_idx = train_test_split(
                outer_train_idx, test_size=outer_es_val_frac, random_state=seed, shuffle=True
            )
        else:
            tr_full_idx = outer_train_idx

        train_loader_full = make_loader(X, tr_full_idx, batch_size=c["batch_size"], shuffle=True)
        if use_outer_val:
            outer_val_loader = make_loader(X, outer_val_idx, batch_size=c["batch_size"], shuffle=False)

        model = ScVIModel(
            n_input=input_dim,
            n_hidden=c["n_hidden"],
            n_latent=c["n_latent"],
            n_layers=c["n_layers"],
            dropout_rate=0.1,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=c["lr"])

        if early_stop:
            es_outer = EarlyStopping(patience=patience, min_delta=min_delta)
        for ep in range(1, epochs_outer + 1):
            tr_loss = train_one_epoch(
                model, train_loader_full, optimizer, device, c["beta"],
                weight_strategy, c["zero_weight"], c["nonzero_weight"], c["trans1"],
            )
            if early_stop and (ep % check_every == 0):
                if use_outer_val:
                    monitor = eval_loss(
                        model, outer_val_loader, device, c["beta"],
                        weight_strategy, c["zero_weight"], c["nonzero_weight"], c["trans1"])
                else:
                    monitor = tr_loss
                if es_outer.step(monitor, model):
                    tag = "val" if use_outer_val else "train"
                    print(f"[outer retrain] early-stopped at epoch {ep}, "
                          f"best_{tag}={es_outer.best:.4f}", flush=True)
                    if es_outer.best_state is not None:
                        model.load_state_dict(es_outer.best_state)
                    break

        # test evaluation
        test_loader = make_loader(X, outer_test_idx, batch_size=c["batch_size"], shuffle=False)
        test_loss = eval_loss(
            model, test_loader, device, c["beta"],
            weight_strategy, c["zero_weight"], c["nonzero_weight"], c["trans1"])
        test_metric = test_loss

        fold_test_losses.append(test_loss)
        fold_test_metrics.append(test_metric)
        print(f"[Test] loss={test_loss:.4f}, metric({eval_metric})={test_metric:.4f}")

        # save
        fold_dir = os.path.join(save_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, "scvi_mse_weights.pt"))
        with open(os.path.join(fold_dir, "scvi_mse_config.json"), "w") as f:
            json.dump({
                "input_dim": int(input_dim),
                "threshold": float(c["threshold"]),
                "trans1": str(c["trans1"]),
                "n_hidden": int(c["n_hidden"]),
                "n_latent": int(c["n_latent"]),
                "n_layers": int(c["n_layers"]),
                "lr": float(c["lr"]),
                "batch_size": int(c["batch_size"]),
                "beta": float(c["beta"]),
                "zero_weight": float(c["zero_weight"]),
                "nonzero_weight": float(c["nonzero_weight"]),
                "weight_strategy": weight_strategy,
                "eval_metric": eval_metric,
                "inner_val_loss": float(best_val_loss),
                "inner_val_metric": float(best_val),
                "outer_test_loss": float(test_loss),
                "outer_test_metric": float(test_metric),
                "seed": int(seed),
            }, f)
        print(f"[SAVE] saved to: {fold_dir}")

    # summary
    val_df = pd.DataFrame(all_val_results)
    val_df_grouped = val_df.groupby('config_name')['val_metric'].mean().reset_index()
    val_df_grouped.columns = ['config_name', 'mean_val_metric']
    val_df_grouped = val_df_grouped.sort_values('mean_val_metric')
    val_path = os.path.join(save_dir, 'all_validation_results.csv')
    val_df_grouped.to_csv(val_path, index=False)
    print(f"\n[SAVE] All validation results saved to: {val_path}")
    print(f"\n{'=' * 60}")
    print("TOP 5 CONFIGURATIONS BY VALIDATION METRIC:")
    print(val_df_grouped.head())

    mean_test = float(np.mean(fold_test_losses))
    std_test = float(np.std(fold_test_losses, ddof=1)) if len(fold_test_losses) > 1 else 0.0
    mean_test_metric = float(np.mean(fold_test_metrics))
    std_test_metric = float(np.std(fold_test_metrics, ddof=1)) if len(fold_test_metrics) > 1 else 0.0

    print(f"\n[FINAL] {n_splits}-fold Test Loss: mean={mean_test:.4f}, sd={std_test:.4f}")
    print(f"[FINAL] {n_splits}-fold Test Metric ({eval_metric}): "
          f"mean={mean_test_metric:.4f}, sd={std_test_metric:.4f}")

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
        "validation_df": val_df_grouped,
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    df1_raw = pd.read_feather(args.data_path1)

    hidden_grid = parse_grid(args.hidden_grid, int)
    latent_grid = parse_grid(args.latent_grid, int)
    n_layers_grid = parse_grid(args.n_layers_grid, int)
    lr_grid = [float(x) for x in args.lr_grid.split(",") if x.strip()]
    bs_grid = parse_grid(args.batch_size_grid, int)
    beta_grid = [float(x) for x in args.beta_grid.split(",") if x.strip()]
    zero_weight_grid = [float(x) for x in args.zero_weight_grid.split(",") if x.strip()]
    nonzero_weight_grid = [float(x) for x in args.nonzero_weight_grid.split(",") if x.strip()]
    threshold_grid = [float(x) for x in args.threshold_grid.split(",") if x.strip()]
    trans1_grid = [x.strip() for x in args.trans1_grid.split(",") if x.strip()]

    save_dir = os.path.join(os.path.dirname(args.out_summary), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    results = outer10_inner_holdout(
        df1_raw=df1_raw,
        device=device,
        hidden_grid=hidden_grid,
        latent_grid=latent_grid,
        n_layers_grid=n_layers_grid,
        lr_grid=lr_grid,
        bs_grid=bs_grid,
        beta_grid=beta_grid,
        threshold_grid=threshold_grid,
        trans1_grid=trans1_grid,
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
        eval_metric=args.eval_metric,
    )

    if args.out_summary:
        out_dir = os.path.dirname(args.out_summary)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_summary, "w") as f:
            f.write(f"# Evaluation metric: {args.eval_metric}\n")
            f.write("fold\tthreshold\ttrans1\tn_hidden\tn_latent\tn_layers\t"
                    "lr\tbatch_size\tbeta\tzero_weight\tnonzero_weight\t"
                    "inner_val_loss\tinner_val_metric\ttest_loss\ttest_metric\n")
            for i, (cfg, vl, vm, tl, tm) in enumerate(zip(
                results["best_cfgs_per_fold"],
                results["val_losses_per_fold"],
                results["val_metrics_per_fold"],
                results["test_losses_per_fold"],
                results["test_metrics_per_fold"],
            ), 1):
                f.write(f"{i}\t{cfg['threshold']}\t{cfg['trans1']}\t"
                        f"{cfg['n_hidden']}\t{cfg['n_latent']}\t{cfg['n_layers']}\t"
                        f"{cfg['lr']}\t{cfg['batch_size']}\t{cfg['beta']}\t"
                        f"{cfg['zero_weight']}\t{cfg['nonzero_weight']}\t"
                        f"{vl:.6f}\t{vm:.6f}\t{tl:.6f}\t{tm:.6f}\n")
            f.write(f"# mean_test_loss\t{results['test_loss_mean']:.6f}\n")
            f.write(f"# sd_test_loss\t{results['test_loss_sd']:.6f}\n")
            f.write(f"# mean_test_metric ({args.eval_metric})\t{results['test_metric_mean']:.6f}\n")
            f.write(f"# sd_test_metric ({args.eval_metric})\t{results['test_metric_sd']:.6f}\n")
        print(f"[SAVE] wrote summary to {args.out_summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="scVI-structure with Weighted MSE Loss (standalone, single input)")
    parser.add_argument('--data_path1', type=str, required=True)
    parser.add_argument('--threshold_grid', type=str, default="1")
    parser.add_argument('--trans1_grid', type=str, default="sqrt+1,log2,sqrt,no_trans")
    parser.add_argument('--hidden_grid', type=str, default="128",
                        help="n_hidden per layer (scVI uses same width for all layers)")
    parser.add_argument('--latent_grid', type=str, default="10")
    parser.add_argument('--n_layers_grid', type=str, default="1",
                        help="Number of hidden layers in encoder & decoder FCLayers")
    parser.add_argument('--lr_grid', type=str, default="0.0001")
    parser.add_argument('--batch_size_grid', type=str, default="128")
    parser.add_argument('--beta_grid', type=str, default="0,0.5,1.0")
    parser.add_argument('--weight_strategy', type=str, default='fixed',
                        choices=['fixed', 'sparsity_aware', 'magnitude', 'focal'])
    parser.add_argument('--zero_weight_grid', type=str, default="1.0")
    parser.add_argument('--nonzero_weight_grid', type=str, default="1.0,5.0,10.0,20.0")
    parser.add_argument('--eval_metric', type=str, default='val_loss',
                        choices=['val_loss'])
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