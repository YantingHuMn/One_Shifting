"""
Shared implementation for the two MLP tasks.

Both tasks train ONE multi-output MLP that predicts every target gene at once: the hidden
layers are shared and the final layer has output_dim = number of target genes. They differ
only in which matrices prep_mlp_data.R writes, so the training and prediction code is the
same and lives here; Step2_train_MLP{,_cross}.py and Step3_predict_MLP{,_cross}.py are the
two command-line entry points.

  task "within" : TF features of one modality  -> non-TF HVGs of the same modality
  task "cross"  : all ATAC gene-activity features -> non-TF HVGs of the paired RNA matrix

Every matrix carries a `pos` column of cell identifiers. Inputs and targets are paired by
that name, never by row position, and prediction re-orders the input columns to the gene
order recorded at training time. A mismatch raises instead of being silently tolerated.
"""

import json
import os
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

ID_COLS = ("pos", "index", "Unnamed: 0", "cell", "cell_id", "barcode")


# --------------------------------------------------------------------- model
class MLP(nn.Module):
    """Multi-output MLP: shared hidden layers, one output unit per target gene.

    The output layer is followed by a ReLU, so every prediction is non-negative in target
    space. Zero is the attainable minimum for count-like targets (no_trans, sqrt, log2,
    log1p) but lies below the attainable minimum of shifted transforms (under sqrt+1 no
    target is below 1.0), and it is wrong for targets that can be negative (standardize).
    An output unit that sits in the negative region receives no gradient.
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return F.relu(self.fc3(h))


# ---------------------------------------------------------------- transforms
def apply_trans(arr, trans):
    a = np.asarray(arr, dtype=np.float64)
    c = np.clip(a, 0, None)
    if trans == "no_trans":            return a.copy()
    if trans == "sqrt":                return np.sqrt(c)
    if trans == "sqrt+1":              return np.sqrt(c + 1)
    if trans == "sqrt+0.00001":        return np.sqrt(c + 0.00001)
    if trans == "sqrt+10":             return np.sqrt(c + 10)
    if trans == "sqrt+1_then_minus_1": return np.sqrt(c + 1) - 1
    if trans == "log2":                return np.log2(c + 1)
    if trans == "log1p":               return np.log1p(c)
    if trans == "log2_then_add_1":     return np.log2(c + 1) + 1
    if trans == "count+1":             return a + 1
    if trans == "log2(count+2)":       return np.log2(c + 2)
    if trans == "log2(count+1)+1":     return np.log2(c + 1) + 1
    raise ValueError(f"Unknown transformation: {trans}")


def inverse_trans(arr, trans):
    """Map back to the pre-transformation scale.

    The model output is non-negative (final ReLU) but can still fall below the range the
    forward transform can produce: under sqrt+1 the attainable minimum is 1.0, so an output
    in [0, 1) inverts to a negative value (a ** 2 - 1). The exponent is clipped so the
    inverse cannot overflow, and the result is returned as-is rather than floored --
    flooring would hide how far outside the attainable range a prediction landed.

    This undoes the TRANSFORMATION only. Any per-cell library-size normalization is undone
    separately (see inverse_norm).
    """
    a = np.asarray(arr, dtype=np.float64)
    p2 = lambda z: np.power(2.0, np.clip(z, -60.0, 60.0))
    if trans == "no_trans":            return a.copy()
    if trans == "sqrt":                return a ** 2
    if trans == "sqrt+1":              return a ** 2 - 1
    if trans == "sqrt+0.00001":        return a ** 2 - 0.00001
    if trans == "sqrt+10":             return a ** 2 - 10
    if trans == "sqrt+1_then_minus_1": return (a + 1) ** 2 - 1
    if trans == "log2":                return p2(a) - 1
    if trans == "log1p":               return np.exp(np.clip(a, -60.0, 60.0)) - 1
    if trans == "log2_then_add_1":     return p2(a - 1) - 1
    if trans == "count+1":             return a - 1
    if trans == "log2(count+2)":       return p2(a) - 2
    if trans == "log2(count+1)+1":     return p2(a - 1) - 1
    raise ValueError(f"Unknown transformation: {trans}")


# ------------------------------------------------------- inverse normalization
def parse_norm_scale(norm):
    """None for "no_norm"; the positive scale s for a numeric library-size normalization.

    Forward normalization (R/normalize_and_save.R):  x_norm = count / lib_size * s.
    Anything else ("standardize", "maximum", a non-numeric string, a non-positive or
    non-finite number) cannot be undone from lib_sizes.csv alone and raises instead of
    being treated as no_norm.
    """
    if norm == "no_norm":
        return None
    try:
        s = float(norm)
    except (TypeError, ValueError):
        raise ValueError(
            f"Cannot invert target normalization '{norm}': only 'no_norm' or a positive "
            f"numeric library-size scale is supported.")
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Target normalization scale must be finite and > 0, got '{norm}'")
    return s


def inverse_norm(arr, norm, lib_sizes=None):
    """Undo per-cell library-size normalization:  count = x_norm * lib_size / s.

    `arr` is cells x genes, already back on the pre-transformation scale (i.e. after
    inverse_trans). `lib_sizes` is one value per row of `arr`: the library size of the
    TARGET modality's original matrix, not of the filtered TF/HVG sub-matrix.
    """
    s = parse_norm_scale(norm)
    a = np.asarray(arr, dtype=np.float64)
    if s is None:
        return a.copy()
    if lib_sizes is None:
        raise ValueError(f"target_norm '{norm}' needs per-cell library sizes (--target_lib_sizes)")
    L = np.asarray(lib_sizes, dtype=np.float64)
    if L.shape != (a.shape[0],):
        raise ValueError(f"lib_sizes has shape {L.shape}, expected ({a.shape[0]},)")
    return a * L[:, None] / s


def read_lib_sizes(path, cell_ids):
    """Library size per cell, matched to `cell_ids` by NAME, not by row position.

    Expects the file written by normalize_and_save.R: columns `cell` and `lib_size`. Rows
    may be in any order. Raises on a duplicated id anywhere in the file, on any requested
    cell that is absent, and on any requested cell whose library size is non-numeric,
    non-finite or <= 0.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"target lib_sizes file not found: {path}")
    df = pd.read_csv(path, dtype={"cell": str})
    for col in ("cell", "lib_size"):
        if col not in df.columns:
            raise ValueError(f"{path} must have columns 'cell' and 'lib_size'; "
                             f"found {list(df.columns)}")
    dup = df.loc[df["cell"].duplicated(keep=False), "cell"].unique()
    if len(dup):
        raise ValueError(f"{path} has {len(dup)} duplicated cell id(s) (e.g. {list(dup[:3])})")
    lib = pd.Series(pd.to_numeric(df["lib_size"], errors="coerce").to_numpy(dtype=np.float64),
                    index=df["cell"].to_numpy())
    ids = [str(c) for c in cell_ids]
    absent = [c for c in ids if c not in lib.index]
    if absent:
        raise ValueError(f"{len(absent)} test cell(s) missing from {path} "
                         f"(e.g. {absent[:3]})")
    out = lib.loc[ids].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(out) | (out <= 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} test cell(s) have a non-finite or non-positive "
                         f"library size in {path} (e.g. {[ids[i] for i in np.where(bad)[0][:3]]})")
    return out


def read_raw_truth(path, cell_ids, genes):
    """Raw-count truth for `cell_ids` x `genes`, in exactly that cell and gene order."""
    rid, R = read_matrix(path, "target raw matrix")
    ids = [str(c) for c in cell_ids]
    pos = pd.Series(np.arange(len(rid)), index=rid)
    absent = [c for c in ids if c not in pos.index]
    if absent:
        raise ValueError(f"{len(absent)} test cell(s) missing from the raw target matrix "
                         f"{path} (e.g. {absent[:3]})")
    absent_g = [g for g in genes if g not in R.columns]
    if absent_g:
        raise ValueError(f"{len(absent_g)} target gene(s) missing from the raw target matrix "
                         f"{path} (e.g. {absent_g[:3]})")
    raw = R[list(genes)].to_numpy(dtype=np.float64)[pos.loc[ids].to_numpy()]
    if not np.isfinite(raw).all():
        raise ValueError(f"non-finite values in the raw target matrix {path}")
    return raw


def within_raw_counts(args, meta, cell_ids, genes, target_norm_df, pred_target_space):
    """Within task: (pred_raw, truth_raw), both on the raw-count scale.

    pred_raw = inverse_trans(pred_target_space) * L / s   (no extra scaling for no_norm)
    truth_raw is read from --target_raw_matrix, not reconstructed from the normalized truth.
    As a guard against a wrong lib_sizes file or raw matrix, raw truth is pushed forward
    through the normalization and must reproduce the normalized target the model was
    trained on.
    """
    if not args.target_raw_matrix:
        raise ValueError("task 'within' requires --target_raw_matrix")
    s = parse_norm_scale(meta["target_norm"])
    if s is not None and not args.target_lib_sizes:
        raise ValueError(f"target_norm '{meta['target_norm']}' requires --target_lib_sizes")
    truth_raw = read_raw_truth(args.target_raw_matrix, cell_ids, genes)
    lib = read_lib_sizes(args.target_lib_sizes, cell_ids) if s is not None else None

    expected = truth_raw if s is None else truth_raw / lib[:, None] * s
    if not np.allclose(expected, target_norm_df.to_numpy(dtype=np.float64),
                       rtol=1e-4, atol=1e-6):
        raise ValueError(
            "raw target matrix and lib_sizes do not reproduce the normalized target "
            f"(target_norm='{meta['target_norm']}'); refusing to write raw-scale output. "
            "Check that --target_raw_matrix / --target_lib_sizes belong to the same modality "
            "and gene filter as the matrix the model was trained on.")

    pred_raw = inverse_norm(inverse_trans(pred_target_space, meta["target_trans"]),
                            meta["target_norm"], lib)
    return pred_raw, truth_raw


# ------------------------------------------------------------------- loading
def read_matrix(path, label):
    """Return (cell_ids, DataFrame of gene columns)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_feather(path)
    hit = [c for c in df.columns if c in ID_COLS]
    if not hit:
        raise ValueError(
            f"{label} ({path}) has no cell-identifier column; one of {ID_COLS} is "
            f"required because inputs and targets are paired by cell name."
        )
    ids = df[hit[0]].astype(str).to_numpy()
    df = df.drop(columns=[c for c in df.columns if c in ID_COLS])
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} has duplicate cell ids")
    return ids, df


def read_pair(input_path, target_path, label):
    """Read an input/target pair and require identical cell ids in identical order."""
    xid, X = read_matrix(input_path, f"{label} input")
    yid, Y = read_matrix(target_path, f"{label} target")
    if len(xid) != len(yid) or not np.array_equal(xid, yid):
        common = set(xid) & set(yid)
        raise ValueError(
            f"{label}: input and target cell ids do not match "
            f"({len(xid)} vs {len(yid)} cells, {len(common)} shared). "
            f"Re-run prep_mlp_data.R --mode extract so both files come from one alignment."
        )
    return xid, X, Y


def read_gene_list(path):
    with open(path) as fh:
        return [g.strip() for g in fh if g.strip()]


# ------------------------------------------------------------------ training
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.bad, self.stopped = float("inf"), 0, 0, False

    def step(self, metric, epoch):
        if metric < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad = metric, int(epoch), 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.stopped = True
        return self.stopped


def _epoch(model, loader, optimizer, device):
    model.train()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        tot += loss.item() * x.size(0); n += x.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        tot += F.mse_loss(model(x), y).item() * x.size(0); n += x.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    model.eval()
    out = []
    for (xb,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch_size, shuffle=False):
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def _fit(cfg, Xtr, Ytr, device, epochs, seed, dropout, Xva=None, Yva=None,
         patience=10, min_delta=0.0):
    """Train one configuration from a fresh, identically seeded initialisation.

    Validation given -> early-stop on it and report the epoch at which the best
    validation loss was reached. Validation absent -> train exactly `epochs` epochs with
    no early stopping; this is the pass that produces the model actually kept.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(Xtr.shape[1], cfg["h1"], cfg["h2"], Ytr.shape[1], dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    tl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(Ytr, dtype=torch.float32)),
                    batch_size=cfg["bs"], shuffle=True)
    if Xva is None:
        for _ in range(int(epochs)):
            _epoch(model, tl, opt, device)
        return model, int(epochs), None
    vl = DataLoader(TensorDataset(torch.tensor(Xva, dtype=torch.float32),
                                  torch.tensor(Yva, dtype=torch.float32)),
                    batch_size=cfg["bs"], shuffle=False)
    es = EarlyStopping(patience, min_delta)
    for ep in range(1, int(epochs) + 1):
        _epoch(model, tl, opt, device)
        if es.step(_eval(model, vl, device), ep):
            break
    return model, max(es.best_epoch, 1), es.best


# ------------------------------------------------------------------- metrics
def per_gene_metrics(pred, truth, genes):
    """Pearson, Spearman and MSE for every gene, with one rule for degenerate genes.

    A correlation is undefined when either vector is constant. Such genes are reported as
    NaN and counted rather than dropped, so a summary cannot be inflated by quietly
    averaging over the subset that happened to work.
    """
    rows = []
    for j, g in enumerate(genes):
        p = np.asarray(pred[:, j], dtype=np.float64)
        t = np.asarray(truth[:, j], dtype=np.float64)
        ok = np.isfinite(p) & np.isfinite(t)
        mse = float(np.mean((p[ok] - t[ok]) ** 2)) if ok.any() else np.nan
        pc = np.std(p[ok]) < 1e-12 if ok.any() else True
        tc = np.std(t[ok]) < 1e-12 if ok.any() else True
        if ok.sum() < 3 or pc or tc:
            pr = sr = np.nan
        else:
            pr = pearsonr(p[ok], t[ok])[0]
            sr = spearmanr(p[ok], t[ok])[0]
        rows.append(dict(gene=g, pearson=pr, spearman=sr, mse=mse,
                         n_cells=int(ok.sum()),
                         pred_constant=bool(pc), truth_constant=bool(tc)))
    return pd.DataFrame(rows)


def summarize(df, space):
    """Collapse a per-gene table, reporting the denominators rather than hiding them."""
    n = len(df)
    valid = df["pearson"].notna()
    return {
        "space": space,
        "n_target_genes": int(n),
        "n_defined": int(valid.sum()),
        "n_undefined": int((~valid).sum()),
        "n_pred_constant": int(df["pred_constant"].sum()),
        "n_truth_constant": int(df["truth_constant"].sum()),
        "mean_pearson_defined":  float(df.loc[valid, "pearson"].mean())  if valid.any() else np.nan,
        "mean_spearman_defined": float(df.loc[valid, "spearman"].mean()) if valid.any() else np.nan,
        "mean_pearson_all_undefined_as_zero":  float(df["pearson"].fillna(0.0).mean()),
        "mean_spearman_all_undefined_as_zero": float(df["spearman"].fillna(0.0).mean()),
        "mean_mse": float(df["mse"].mean(skipna=True)),
    }


# ------------------------------------------------------------- entry points
def run_training(args, task):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tr_id, Xtr_df, Ytr_df = read_pair(args.input_train, args.target_train, "train")
    va_id, Xva_df, Yva_df = read_pair(args.input_val,   args.target_val,   "val")
    if set(tr_id) & set(va_id):
        raise ValueError("train and validation share cells; the split is not disjoint")

    input_genes  = list(Xtr_df.columns)
    target_genes = list(Ytr_df.columns)
    if list(Xva_df.columns) != input_genes or list(Yva_df.columns) != target_genes:
        raise ValueError("train and validation gene columns differ")

    Xtr = apply_trans(Xtr_df.to_numpy(), args.input_trans).astype(np.float32)
    Xva = apply_trans(Xva_df.to_numpy(), args.input_trans).astype(np.float32)
    Ytr = apply_trans(Ytr_df.to_numpy(), args.target_trans).astype(np.float32)
    Yva = apply_trans(Yva_df.to_numpy(), args.target_trans).astype(np.float32)
    for nm, a in (("input", Xtr), ("target", Ytr)):
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite values in the transformed {nm} training matrix")

    grid = [dict(h1=h1, h2=h2, lr=lr, bs=bs) for h1, h2, lr, bs in product(
        [int(v) for v in args.hidden_grid1.split(",") if v.strip()],
        [int(v) for v in args.hidden_grid2.split(",") if v.strip()],
        [float(v) for v in args.lr_grid.split(",") if v.strip()],
        [int(v) for v in args.batch_size_grid.split(",") if v.strip()])]
    print(f"Device: {device}")
    print(f"[PLAN] task={task}  {len(input_genes)} input features -> {len(target_genes)} "
          f"target genes;  {len(tr_id)} train / {len(va_id)} val cells;  "
          f"{len(grid)} configuration(s)")

    # pass 1 -- configuration and epoch count, chosen on validation only
    best = None
    for cfg in grid:
        _, ep, vloss = _fit(cfg, Xtr, Ytr, device, args.epochs, args.seed, args.dropout,
                            Xva, Yva, args.patience, args.min_delta)
        print(f"  cfg h1={cfg['h1']} h2={cfg['h2']} lr={cfg['lr']} bs={cfg['bs']}"
              f"  best_epoch={ep}  val_mse={vloss:.6f}")
        if vloss is not None and np.isfinite(vloss) and (best is None or vloss < best[2]):
            best = (cfg, ep, vloss)
    if best is None:
        raise RuntimeError("every configuration produced a non-finite validation loss")
    cfg, best_epoch, val_mse = best
    print(f"[PASS 1] selected {cfg}  best_epoch={best_epoch}  val_mse={val_mse:.6f}")

    # pass 2 -- retrain from scratch on train + validation for exactly that many epochs
    Xall = np.concatenate([Xtr, Xva], axis=0)
    Yall = np.concatenate([Ytr, Yva], axis=0)
    model, _, _ = _fit(cfg, Xall, Yall, device, best_epoch, args.seed, args.dropout)
    print(f"[PASS 2] retrained on all {Xall.shape[0]} train+val cells for {best_epoch} epochs")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model.pt"))
    meta = dict(
        task=task, architecture="MLP", output_activation="relu", input_dim=len(input_genes),
        output_dim=len(target_genes), hidden_dim1=cfg["h1"], hidden_dim2=cfg["h2"],
        dropout=args.dropout, lr=cfg["lr"], batch_size=cfg["bs"],
        best_epoch=int(best_epoch), epoch_cap=int(args.epochs),
        best_epoch_hit_cap=bool(best_epoch >= args.epochs),
        val_mse=float(val_mse), seed=int(args.seed),
        input_trans=args.input_trans, target_trans=args.target_trans,
        input_norm=args.input_norm, target_norm=args.target_norm,
        input_genes=input_genes, target_genes=target_genes,
        train_cells=[str(c) for c in tr_id], val_cells=[str(c) for c in va_id],
    )
    with open(os.path.join(args.model_dir, "final_config.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if args.out_summary:
        os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
        pd.DataFrame([{k: v for k, v in meta.items()
                       if not isinstance(v, list)}]).to_csv(args.out_summary,
                                                            sep="\t", index=False)
    print(f"[SAVE] {os.path.join(args.model_dir, 'final_model.pt')}")
    print(f"[SAVE] {os.path.join(args.model_dir, 'final_config.json')}")


def run_prediction(args, task):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg_path = os.path.join(args.model_dir, "final_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found -- training did not finish for this combination. "
            f"Refusing to write a partial result.")
    with open(cfg_path) as fh:
        meta = json.load(fh)
    if meta.get("task") != task:
        raise ValueError(f"model was trained for task '{meta.get('task')}', not '{task}'")
    if meta.get("output_activation") != "relu":
        raise ValueError(
            f"{cfg_path} has no output_activation='relu': this model was trained with the "
            f"old linear output layer. The weights load without error but the network is a "
            f"different function now -- retrain into a new output directory.")

    te_id, X_df, Y_df = read_pair(args.input_test, args.target_test, "test")
    seen = set(meta["train_cells"]) | set(meta["val_cells"])
    leaked = sorted(set(map(str, te_id)) & seen)
    if leaked:
        raise ValueError(f"{len(leaked)} test cells were used in training "
                         f"(e.g. {leaked[:3]}) -- the split is broken")

    missing = [g for g in meta["input_genes"] if g not in X_df.columns]
    if missing:
        raise ValueError(f"{len(missing)} input genes the model expects are absent from "
                         f"the test matrix (e.g. {missing[:3]})")
    X_df = X_df[meta["input_genes"]]                      # re-order by name, not position
    missing_t = [g for g in meta["target_genes"] if g not in Y_df.columns]
    if missing_t:
        raise ValueError(f"{len(missing_t)} target genes are absent from the test truth "
                         f"matrix (e.g. {missing_t[:3]})")
    Y_df = Y_df[meta["target_genes"]]

    Xt = apply_trans(X_df.to_numpy(), meta["input_trans"]).astype(np.float32)
    pred_t = predict(model_from(meta, args.model_dir, device), Xt, device)
    truth_t = apply_trans(Y_df.to_numpy(), meta["target_trans"])
    genes = meta["target_genes"]
    if task == "within":
        # raw-count scale: undo the transformation AND the library-size normalization
        pred_o, truth_o = within_raw_counts(args, meta, te_id, genes, Y_df, pred_t)
        scale_tag, space_t, space_o = "raw_counts", "target_space", "raw_counts"
    else:
        pred_o = inverse_trans(pred_t, meta["target_trans"])
        truth_o = Y_df.to_numpy(dtype=np.float64)
        scale_tag, space_t, space_o = ("original_scale",
                                       f"target_trans:{meta['target_trans']}", "original_scale")

    os.makedirs(args.out_dir, exist_ok=True)
    def _write(arr, name):
        d = pd.DataFrame(np.asarray(arr, dtype=np.float64), columns=genes)
        d.insert(0, "pos", te_id)
        d.to_feather(os.path.join(args.out_dir, name))
    _write(pred_t,  "pred_target_space.feather")
    _write(truth_t, "truth_target_space.feather")
    _write(pred_o,  f"pred_{scale_tag}.feather")
    _write(truth_o, f"truth_{scale_tag}.feather")

    m_t = per_gene_metrics(np.asarray(pred_t),  np.asarray(truth_t),  genes)
    m_o = per_gene_metrics(np.asarray(pred_o),  np.asarray(truth_o),  genes)
    m_t["space"] = space_t
    m_o["space"] = space_o
    pd.concat([m_t, m_o], ignore_index=True).to_csv(
        os.path.join(args.out_dir, "per_gene_metrics.csv"), index=False)

    rows = []
    for df_, space in ((m_t, space_t), (m_o, space_o)):
        r = summarize(df_, space)
        r.update(task=task, method=args.method_label,
                 input_trans=meta["input_trans"], target_trans=meta["target_trans"],
                 input_norm=meta["input_norm"], target_norm=meta["target_norm"],
                 seed=meta["seed"], n_test_cells=int(len(te_id)))
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.out_dir, "summary_metrics.csv"), index=False)

    if args.compare_path:
        keys = ["method", "task", "space", "input_trans", "target_trans",
                "input_norm", "target_norm"]
        os.makedirs(os.path.dirname(args.compare_path) or ".", exist_ok=True)
        if os.path.exists(args.compare_path):
            old = pd.read_csv(args.compare_path)
            if set(old.columns) != set(summ.columns):
                raise ValueError(
                    f"{args.compare_path} was written with a different column set "
                    f"(missing: {sorted(set(summ.columns) - set(old.columns))}); "
                    f"delete it and re-run.")
            out = pd.concat([old[summ.columns], summ], ignore_index=True)
            out = out[~out.duplicated(subset=keys, keep="last")]
        else:
            out = summ
        out.to_csv(args.compare_path, index=False)

    for r in rows:
        print(f"[{r['space']}] genes={r['n_target_genes']} defined={r['n_defined']} "
              f"undefined={r['n_undefined']} (pred_constant={r['n_pred_constant']}, "
              f"truth_constant={r['n_truth_constant']})  "
              f"mean_pearson_defined={r['mean_pearson_defined']:.6f}  "
              f"mean_pearson_all={r['mean_pearson_all_undefined_as_zero']:.6f}")
    print(f"[SAVE] {args.out_dir}")


def model_from(meta, model_dir, device):
    m = MLP(meta["input_dim"], meta["hidden_dim1"], meta["hidden_dim2"],
            meta["output_dim"], dropout=0.0).to(device)
    m.load_state_dict(torch.load(os.path.join(model_dir, "final_model.pt"),
                                 map_location=device, weights_only=True))
    m.eval()
    return m


# ----------------------------------------------------------------- CLI parts
def add_train_args(p):
    p.add_argument("--input_train", required=True)
    p.add_argument("--input_val", required=True)
    p.add_argument("--target_train", required=True)
    p.add_argument("--target_val", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out_summary", default=None)
    p.add_argument("--input_trans", default="no_trans")
    p.add_argument("--target_trans", default="no_trans")
    p.add_argument("--input_norm", default="no_norm")
    p.add_argument("--target_norm", default="no_norm")
    p.add_argument("--hidden_grid1", default="256")
    p.add_argument("--hidden_grid2", default="128")
    p.add_argument("--lr_grid", default="0.0001")
    p.add_argument("--batch_size_grid", default="64")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p


def add_predict_args(p, default_label):
    p.add_argument("--model_dir", required=True)
    p.add_argument("--input_test", required=True)
    p.add_argument("--target_test", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--compare_path", default=None)
    # Within task only: raw-count truth and per-cell library sizes of the TARGET modality,
    # used to bring predictions back to raw counts. Cross does not need them.
    p.add_argument("--target_raw_matrix", default=None)
    p.add_argument("--target_lib_sizes", default=None)
    p.add_argument("--method_label", default=default_label)
    p.add_argument("--cpu", action="store_true")
    return p