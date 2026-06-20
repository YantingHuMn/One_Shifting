"""
MLP: Predict target gene expression using TF expression (per gene)
============================================================

Input: 6 pre-split feather files
  --tf_train      (n_train x n_TFs)
  --tf_val        (n_val   x n_TFs)
  --tf_test       (n_test  x n_TFs)
  --target_train  (n_train x n_genes)
  --target_val    (n_val   x n_genes)
  --target_test   (n_test  x n_genes)

Training procedure (for each target gene):
  1. Train on train set, use val for early stopping + hyperparameter search
  2. After selecting best hyperparameters, retrain on train+val combined
  3. Final evaluation on test set
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import argparse
import os
import numpy as np
from itertools import product
from scipy.stats import pearsonr, spearmanr
import json

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h)


def weighted_mse_loss(pred, target, weight_strategy='none',
                      zero_weight=1.0, nonzero_weight=5.0):
    if weight_strategy == 'fixed':
        zero_mask = (torch.abs(target) < 1e-8).float()
        nonzero_mask = 1.0 - zero_mask
        weights = zero_mask * zero_weight + nonzero_mask * nonzero_weight
    else:
        weights = torch.ones_like(target)
    return torch.mean(weights * (pred - target) ** 2)


def train_one_epoch(model, loader, optimizer, device, weight_strategy='none'):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = weighted_mse_loss(pred, y, weight_strategy)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, weight_strategy='none'):
    """Return (mse_loss, pearson_r, spearman_r)"""
    model.eval()
    all_pred, all_true = [], []
    total_loss, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = weighted_mse_loss(pred, y, weight_strategy)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    avg_loss = total_loss / max(n, 1)
    pred_arr = np.concatenate(all_pred).flatten()
    true_arr = np.concatenate(all_true).flatten()

    if np.std(pred_arr) < 1e-10 or np.std(true_arr) < 1e-10:
        pcc, scc = 0.0, 0.0
    else:
        pcc, _ = pearsonr(pred_arr, true_arr)
        scc, _ = spearmanr(pred_arr, true_arr)
        pcc = 0.0 if np.isnan(pcc) else pcc
        scc = 0.0 if np.isnan(scc) else scc

    return avg_loss, pcc, scc


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
                self.best_state = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return self.stopped


def _apply_trans(arr, trans):
    arr = np.asarray(arr, dtype=np.float32)

    if trans == "no_trans":
        return arr.copy()
    elif trans == "sqrt":
        return np.sqrt(np.clip(arr, 0, None))
    elif trans == "sqrt+1":
        return np.sqrt(np.clip(arr, 0, None) + 1)
    elif trans == "sqrt+0.00001":
        return np.sqrt(np.clip(arr, 0, None) + 0.00001)
    elif trans == "sqrt+10":
        return np.sqrt(np.clip(arr, 0, None) + 10)
    elif trans == "sqrt+1_then_minus_1":
        return np.sqrt(np.clip(arr, 0, None) + 1) - 1
    elif trans == "log2":
        return np.log2(np.clip(arr, 0, None) + 1)
    elif trans == "log1p":
        return np.log1p(np.clip(arr, 0, None))
    elif trans == "log2_then_add_1":
        return np.log2(np.clip(arr, 0, None) + 1) + 1
    elif trans == "count+1":
        return arr + 1
    elif trans == "log2(count+2)":
        return np.log2(np.clip(arr, 0, None) + 2)
    elif trans == "log2(count+1)+1":
        return np.log2(np.clip(arr, 0, None) + 1) + 1
    else:
        raise ValueError(f"Unknown transformation: {trans}")

def train_one_gene(X_train_np, y_train_np,
                   X_val_np, y_val_np,
                   X_test_np, y_test_np,
                   gene_name, device, args, save_dir):
    """
    For one target gene:
      1. Train on train set, early stopping + hyperparameter selection on val set
      2. Retrain with train+val combined using best config
      3. Final evaluation on test set
    """
    hidden_grid1 = [int(x) for x in args.hidden_grid1.split(",") if x.strip()]
    hidden_grid2 = [int(x) for x in args.hidden_grid2.split(",") if x.strip()]
    lr_grid = [float(x) for x in args.lr_grid.split(",") if x.strip()]
    bs_grid = [int(x) for x in args.batch_size_grid.split(",") if x.strip()]
    trans = args.trans  # fixed value, passed from shell

    best_val_metric = float("inf")
    best_cfg = None
    best_val_pcc = 0.0

    # ---- Step 1: Hyperparameter search (train on train, evaluate on val) ----
    for h1, h2, lr, bs in product(
        hidden_grid1, hidden_grid2, lr_grid, bs_grid
    ):
        X_tr = torch.tensor(_apply_trans(X_train_np, trans), dtype=torch.float32)
        y_tr = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
        X_va = torch.tensor(_apply_trans(X_val_np, trans), dtype=torch.float32)
        y_va = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=bs, shuffle=False)

        model = MLP(X_tr.shape[1], h1, h2, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        es = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

        for ep in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device, args.weight_strategy)
            if ep % args.check_every == 0:
                val_loss, val_pcc, _ = evaluate(model, val_loader, device, args.weight_strategy)
                monitor = -val_pcc if args.eval_metric == 'pearson' else val_loss
                if es.step(monitor, model):
                    break

        if es.best_state is not None:
            model.load_state_dict(es.best_state)

        val_loss, val_pcc, val_scc = evaluate(model, val_loader, device, args.weight_strategy)
        val_metric = -val_pcc if args.eval_metric == 'pearson' else val_loss

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_val_pcc = val_pcc
            best_cfg = dict(trans=trans, h1=h1, h2=h2, lr=lr, bs=bs)

    # Step 2: Retrain with best config on train+val combined
    cfg = best_cfg
    X_trainval = np.concatenate([X_train_np, X_val_np], axis=0)
    y_trainval = np.concatenate([y_train_np, y_val_np], axis=0)

    X_tv = torch.tensor(_apply_trans(X_trainval, cfg['trans']), dtype=torch.float32)
    y_tv = torch.tensor(y_trainval, dtype=torch.float32).unsqueeze(1)

    use_outer_val = (args.outer_es_val_frac > 0.0)
    if use_outer_val:
        n_tv = X_tv.shape[0]
        n_outer_val = round(n_tv * args.outer_es_val_frac)
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_tv)
        outer_val_idx = perm[:n_outer_val]
        train_idx_full = perm[n_outer_val:]
        trainval_loader = DataLoader(
            TensorDataset(X_tv[train_idx_full], y_tv[train_idx_full]),
            batch_size=cfg['bs'],
            shuffle=True
        )
        outer_val_loader = DataLoader(
            TensorDataset(X_tv[outer_val_idx], y_tv[outer_val_idx]),
            batch_size=cfg['bs'],
            shuffle=False
        )
    else:
        trainval_loader = DataLoader(TensorDataset(X_tv, y_tv), batch_size=cfg['bs'], shuffle=True)

    model = MLP(X_tv.shape[1], cfg['h1'], cfg['h2'], dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    es = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, trainval_loader, optimizer, device, args.weight_strategy)
        if ep % args.check_every == 0:
            if use_outer_val:
                val_loss, val_pcc, _ = evaluate(model, outer_val_loader, device, args.weight_strategy)
                monitor = -val_pcc if args.eval_metric == 'pearson' else val_loss
            else:
                monitor = tr_loss
            if es.step(monitor, model):
                tag = "val" if use_outer_val else "train"
                print(f"    [retrain] early-stopped at epoch {ep}, best_{tag}={es.best:.4f}")
                break

    if es.best_state is not None:
        model.load_state_dict(es.best_state)

    # Step 3: Final evaluation on test set
    X_te = torch.tensor(_apply_trans(X_test_np, cfg['trans']), dtype=torch.float32)
    y_te = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=cfg['bs'], shuffle=False)

    test_loss, test_pcc, test_scc = evaluate(model, test_loader, device, args.weight_strategy)

    # Save model 
    gene_dir = os.path.join(save_dir, gene_name)
    os.makedirs(gene_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(gene_dir, "model.pt"))
    with open(os.path.join(gene_dir, "config.json"), "w") as f:
        json.dump({
            'gene': gene_name,
            'input_dim': int(X_tv.shape[1]),
            'val_pearson': float(best_val_pcc),
            'test_loss': float(test_loss),
            'test_pearson': float(test_pcc),
            'test_spearman': float(test_scc),
            **{k: (float(v) if isinstance(v, (int, float)) else str(v))
               for k, v in cfg.items()}
        }, f, indent=2)

    print(f"  Gene={gene_name}: val_PCC={best_val_pcc:.4f}, "
          f"test_loss={test_loss:.4f}, test_PCC={test_pcc:.4f}, "
          f"test_SCC={test_scc:.4f}, cfg={cfg}")

    return {
        'gene': gene_name,
        'val_pearson': best_val_pcc,
        'test_loss': test_loss,
        'test_pearson': test_pcc,
        'test_spearman': test_scc,
        **cfg
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    # ---- Load 6 files ----
    print("[INFO] Loading data...")
    tf_train     = pd.read_feather(args.tf_train)
    tf_val       = pd.read_feather(args.tf_val)
    tf_test      = pd.read_feather(args.tf_test)
    target_train = pd.read_feather(args.target_train)
    target_val   = pd.read_feather(args.target_val)
    target_test  = pd.read_feather(args.target_test)

    # Drop metadata columns
    for col in ['pos', 'index', 'Unnamed: 0']:
        for df in [tf_train, tf_val, tf_test, target_train, target_val, target_test]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    print(f"  TF train:     {tf_train.shape}")
    print(f"  TF val:       {tf_val.shape}")
    print(f"  TF test:      {tf_test.shape}")
    print(f"  Target train: {target_train.shape}")
    print(f"  Target val:   {target_val.shape}")
    print(f"  Target test:  {target_test.shape}")

    assert tf_train.shape[1] == tf_val.shape[1] == tf_test.shape[1], "TF column count mismatch!"
    assert tf_train.shape[0] == target_train.shape[0], "Train cell count mismatch!"
    assert tf_val.shape[0] == target_val.shape[0], "Val cell count mismatch!"
    assert tf_test.shape[0] == target_test.shape[0], "Test cell count mismatch!"

    X_train = tf_train.values.astype(np.float32)
    X_val   = tf_val.values.astype(np.float32)
    X_test  = tf_test.values.astype(np.float32)
    target_train_np = target_train.values.astype(np.float32)
    target_val_np   = target_val.values.astype(np.float32)
    target_test_np  = target_test.values.astype(np.float32)
    gene_names = list(target_train.columns)

    print(f"\nInput dimension: {X_train.shape[1]} TFs")
    print(f"Train: {X_train.shape[0]} cells")
    print(f"Val:   {X_val.shape[0]} cells")
    print(f"Test:  {X_test.shape[0]} cells")
    print(f"Target genes: {len(gene_names)}")

    save_dir = os.path.join(os.path.dirname(args.out_summary), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    # Select target genes
    if args.target_genes:
        selected = [g.strip() for g in args.target_genes.split(",")]
        gene_list = []
        for g in selected:
            if g in gene_names:
                gene_list.append((g, gene_names.index(g)))
            else:
                print(f"  [WARNING] gene '{g}' not found, skipping.")
    else:
        gene_list = [(name, i) for i, name in enumerate(gene_names)]

    print(f"\nWill train MLP for {len(gene_list)} genes...\n")

    # ---- Train per gene ----
    all_results = []

    for gene_name, gene_idx in gene_list:
        print(f"{'='*50}")
        print(f"Gene: {gene_name} ({gene_idx+1}/{len(gene_list)})")

        y_train = target_train_np[:, gene_idx]
        y_val   = target_val_np[:, gene_idx]
        y_test  = target_test_np[:, gene_idx]

        result = train_one_gene(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            gene_name, device, args, save_dir
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.out_summary, index=False, sep='\t')

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Mean val  PCC:  {results_df['val_pearson'].mean():.4f} "
          f"+/- {results_df['val_pearson'].std():.4f}")
    print(f"  Mean test PCC:  {results_df['test_pearson'].mean():.4f} "
          f"+/- {results_df['test_pearson'].std():.4f}")
    print(f"  Mean test SCC:  {results_df['test_spearman'].mean():.4f} "
          f"+/- {results_df['test_spearman'].std():.4f}")
    print(f"  Mean test loss: {results_df['test_loss'].mean():.4f} "
          f"+/- {results_df['test_loss'].std():.4f}")
    print(f"\nResults saved to: {args.out_summary}")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLP: Predict target genes using TFs (train/val/test pre-split)"
    )
    # Data (6 files)
    parser.add_argument('--tf_train', type=str, required=True)
    parser.add_argument('--tf_val', type=str, required=True)
    parser.add_argument('--tf_test', type=str, required=True)
    parser.add_argument('--target_train', type=str, required=True)
    parser.add_argument('--target_val', type=str, required=True)
    parser.add_argument('--target_test', type=str, required=True)
    parser.add_argument('--target_genes', type=str, default=None,
                        help="Comma-separated target gene names; if not specified, predict all")

    # Hyperparameter grid
    parser.add_argument('--trans', type=str, default="no_trans",
                        help="Data transform (passed from shell, e.g. no_trans, sqrt, sqrt+1, log2, count+1, log2(count+2))")
    parser.add_argument('--hidden_grid1', type=str, default="256")
    parser.add_argument('--hidden_grid2', type=str, default="128")
    parser.add_argument('--lr_grid', type=str, default="0.001,0.0001")
    parser.add_argument('--batch_size_grid', type=str, default="64,128")
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--weight_strategy', type=str, default='none',
                        choices=['none', 'fixed', 'sparsity_aware'])
    parser.add_argument('--eval_metric', type=str, default='loss',
                        choices=['loss', 'pearson'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--check_every', type=int, default=1)
    parser.add_argument('--outer_es_val_frac', type=float, default=0.1)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--out_summary', type=str, required=True)

    args = parser.parse_args()
    main(args)