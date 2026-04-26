#!/usr/bin/env python3
import argparse, os, json, glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc5 = nn.Linear(hidden_dim1, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return F.relu(self.fc5(h4))


def apply_transformation(df, trans):
    df_copy = df.copy()

    if trans == "sqrt+1":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:] + 1)
    elif trans == "sqrt":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:])
    elif trans == "log2":
        df_copy.iloc[:, 1:] = np.log2(df_copy.iloc[:, 1:] + 1)
    elif trans == "log2_then_add_1":
        df_copy.iloc[:, 1:] = np.log2(df_copy.iloc[:, 1:] + 1) + 1
    elif trans == "sqrt+0.00001":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:] + 0.00001)
    elif trans == "sqrt+10":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:] + 10)
    elif trans == "sqrt+1_then_minus_1":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:] + 1) - 1
    elif trans == "count+1":
        df_copy.iloc[:, 1:] = df_copy.iloc[:, 1:] + 1
    elif trans == "log2(count+2)":
        df_copy.iloc[:, 1:] = np.log2(df_copy.iloc[:, 1:] + 2)
    elif trans == "log2(count+1)+1":
        df_copy.iloc[:, 1:] = np.log2(df_copy.iloc[:, 1:] + 1) + 1
    elif trans == "no_trans":
        pass
    else:
        raise ValueError(f"Unknown transformation: {trans}")

    return df_copy


def filter_and_transform(df1, threshold_value, trans1, transformed_out_dir, save=False):
    data_cols = df1.columns[1:]
    zero_percentage = (df1[data_cols] == 0).mean()
    keep_cols = zero_percentage < threshold_value

    pos_col = df1.columns[0]
    cols_to_keep = [pos_col] + data_cols[keep_cols].tolist()

    filtered_df1 = df1[cols_to_keep].copy()
    filtered_df1 = apply_transformation(filtered_df1, trans1)

    if save:
        transformed_out_dir = Path(transformed_out_dir)
        transformed_out_dir.mkdir(parents=True, exist_ok=True)
        save_path1 = transformed_out_dir / "Count_matrix_transformed.feather"
        filtered_df1.to_feather(save_path1)
        print(f"[SAVE] Saved transformed df1 ({trans1}) to: {save_path1}")

    return filtered_df1


def find_weights_file(directory):
    for ext in ("*.pt", "*.pth"):
        matches = glob.glob(os.path.join(directory, ext))
        if matches:
            if len(matches) > 1:
                print(f"  [WARNING] Multiple weight files found in {directory}, using: {matches[0]}")
            return matches[0]
    raise FileNotFoundError(f"No .pt or .pth file found in {directory}")


def find_config_file(directory):
    matches = glob.glob(os.path.join(directory, "*.json"))
    if matches:
        if len(matches) > 1:
            print(f"  [WARNING] Multiple JSON files found in {directory}, using: {matches[0]}")
        return matches[0]
    raise FileNotFoundError(f"No .json file found in {directory}")


def find_best_fold(saved_models_dir, criterion="inner_val_loss"):
    fold_dirs = glob.glob(os.path.join(saved_models_dir, "fold_*"))
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {saved_models_dir}")

    best_fold = None
    best_score = float("inf")

    for fold_dir in fold_dirs:
        try:
            config_path = find_config_file(fold_dir)
        except FileNotFoundError:
            print(f"Warning: no JSON config in {fold_dir}, skipping")
            continue

        with open(config_path, "r") as f:
            cfg = json.load(f)

        score = cfg.get(criterion, float("inf"))
        fold_name = os.path.basename(fold_dir)
        print(f"{fold_name}: {criterion}={score:.6f}")

        if score != score:
            if best_fold is None:
                best_fold = fold_dir
        elif score < best_score:
            best_score = score
            best_fold = fold_dir

    if best_fold is None:
        raise ValueError("No valid fold found")

    print(f"\n[BEST] {os.path.basename(best_fold)} with {criterion}={best_score:.6f}")
    return best_fold


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Try final_model first
    final_weights = None
    final_config = None
    for ext in ("*.pt", "*.pth"):
        for f in glob.glob(os.path.join(args.saved_models_dir, ext)):
            if "final" in os.path.basename(f).lower():
                final_weights = f
                break
        if final_weights:
            break
    for f in glob.glob(os.path.join(args.saved_models_dir, "*.json")):
        if "final" in os.path.basename(f).lower():
            final_config = f
            break

    if final_weights and final_config:
        print(f"[INFO] Using final model: {os.path.basename(final_weights)}")
        model_path = final_weights
        config_path = final_config
    else:
        print("[INFO] Final model not found, using best fold")
        fold_dir = find_best_fold(args.saved_models_dir, args.criterion)
        model_path = find_weights_file(fold_dir)
        config_path = find_config_file(fold_dir)
        print(f"  weights: {os.path.basename(model_path)}")
        print(f"  config:  {os.path.basename(config_path)}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Support both "trans1" and "trans" key names
    trans1 = cfg.get("trans1", cfg.get("trans", "no_trans"))

    print(f"\n[CONFIG]")
    print(f"  threshold: {cfg['threshold']}")
    print(f"  trans: {trans1}")
    print(f"  architecture: {cfg['hidden_dim1']}-{cfg['hidden_dim2']}-{cfg['latent_dim']}")

    print("\n[LOAD DATA]")
    df1_raw = pd.read_feather(args.data_path1)
    print(f"  Original df1: {df1_raw.shape}")

    print("\n[PREPROCESS] Applying filter_and_transform...")
    df1_transformed = filter_and_transform(
        df1_raw, cfg['threshold'], trans1,
        transformed_out_dir=args.transformed_out_dir,
        save=True
    )
    print(f"  Transformed df1: {df1_transformed.shape}")

    if 'pos' in df1_transformed.columns:
        pos_col = df1_transformed['pos'].copy()
        df1_transformed = df1_transformed.drop(columns=['pos'])
    else:
        pos_col = None

    X = torch.tensor(df1_transformed.to_numpy(), dtype=torch.float32, device=device)

    input_dim = int(cfg["input_dim"])
    if X.shape[1] != input_dim:
        raise ValueError(f"Dimension mismatch! Expected {input_dim}, got {X.shape[1]}")

    print("\n[MODEL] Loading VAE...")
    model = VAE(input_dim, cfg["hidden_dim1"], cfg["hidden_dim2"], cfg["latent_dim"]).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("[RECONSTRUCT] Running VAE...")
    mu, logvar = model.encode(X)
    Z = mu
    X_recon = model.decode(Z)

    recon_df = pd.DataFrame(X_recon.cpu().numpy(), columns=df1_transformed.columns.tolist())
    if pos_col is not None:
        recon_df.insert(0, 'pos', pos_col.values)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recon_df.to_feather(out_path)

    print(f"\n[DONE] Reconstruction saved to: {out_path}")
    print(f"       Shape: {recon_df.shape}")
    print(f"       Used transformation: {trans1}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VAE Reconstruction / Inference - Single Input")
    ap.add_argument("--data_path1", required=True, help="Input feather file (to reconstruct)")
    ap.add_argument("--transformed_out_dir", required=True, help="Directory to save transformed feather")
    ap.add_argument("--saved_models_dir", required=True, help="saved_models directory")
    ap.add_argument("--criterion", default="inner_val_loss",
                    choices=["inner_val_loss", "outer_test_loss"])
    ap.add_argument("--out_path", required=True, help="Output reconstruction feather path")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)