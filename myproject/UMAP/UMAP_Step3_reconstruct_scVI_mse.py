#!/usr/bin/env python3
import argparse, os, json, glob, collections
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class FCLayers(nn.Module):
    def __init__(self, n_in, n_out, n_cat_list=None, n_layers=1, n_hidden=128,
                 dropout_rate=0.1, use_batch_norm=True, use_layer_norm=False,
                 use_activation=True, bias=True, inject_covariates=True, activation_fn=nn.ReLU):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        if n_cat_list is not None:
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        self.n_cov = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(collections.OrderedDict([
            (f"Layer {i}", nn.Sequential(
                nn.Linear(_n_in + self.n_cov * self._inject_into_layer(i), _n_out, bias=bias),
                nn.BatchNorm1d(_n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                nn.LayerNorm(_n_out, elementwise_affine=False) if use_layer_norm else None,
                activation_fn() if use_activation else None,
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
            ))
            for i, (_n_in, _n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
        ]))

    def _inject_into_layer(self, layer_num):
        return layer_num == 0 or (layer_num > 0 and self.inject_covariates)

    def forward(self, x, *cat_list):
        one_hot_cat_list = []
        cat_list = cat_list or []
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0")
            if n_cat > 1:
                if cat.dim() == 1: cat = cat.unsqueeze(-1)
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


class EncoderSCVI(nn.Module):
    def __init__(self, n_input, n_output, n_cat_list=None, n_layers=1, n_hidden=128,
                 dropout_rate=0.1, use_batch_norm=True, use_layer_norm=False,
                 inject_covariates=True, var_eps=1e-4):
        super().__init__()
        self.var_eps = var_eps
        self.encoder = FCLayers(n_in=n_input, n_out=n_hidden, n_cat_list=n_cat_list,
                                n_layers=n_layers, n_hidden=n_hidden, dropout_rate=dropout_rate,
                                use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm,
                                inject_covariates=inject_covariates)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x, *cat_list):
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + self.var_eps
        dist = torch.distributions.Normal(q_m, q_v.sqrt())
        z = dist.rsample()
        return q_m, q_v, z


class DecoderSCVI(nn.Module):
    def __init__(self, n_input, n_output, n_cat_list=None, n_layers=1, n_hidden=128,
                 inject_covariates=True, use_batch_norm=True, use_layer_norm=False):
        super().__init__()
        self.px_decoder = FCLayers(n_in=n_input, n_out=n_hidden, n_cat_list=n_cat_list,
                                   n_layers=n_layers, n_hidden=n_hidden, dropout_rate=0,
                                   inject_covariates=inject_covariates,
                                   use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm)
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

    def forward(self, z, library, *cat_list):
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_rate = torch.exp(library) * px_scale
        return px_rate


class ScVIModel(nn.Module):
    def __init__(self, n_input, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1,
                 use_batch_norm_encoder=True, use_batch_norm_decoder=True,
                 use_layer_norm_encoder=False, use_layer_norm_decoder=False,
                 log_variational=True):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.z_encoder = EncoderSCVI(n_input=n_input, n_output=n_latent, n_layers=n_layers,
                                     n_hidden=n_hidden, dropout_rate=dropout_rate,
                                     use_batch_norm=use_batch_norm_encoder,
                                     use_layer_norm=use_layer_norm_encoder)
        self.decoder = DecoderSCVI(n_input=n_latent, n_output=n_input, n_layers=n_layers,
                                   n_hidden=n_hidden, inject_covariates=True,
                                   use_batch_norm=use_batch_norm_decoder,
                                   use_layer_norm=use_layer_norm_decoder)

    def forward(self, x):
        library = torch.log(x.sum(dim=1, keepdim=True) + 1e-6)
        x_input = torch.log1p(x) if self.log_variational else x
        q_m, q_v, z = self.z_encoder(x_input)
        px_rate = self.decoder(z, library)
        logvar = torch.log(q_v)
        return px_rate, q_m, logvar


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
    return df


def filter_and_transform(df1, threshold_value, trans1):
    data_cols = df1.columns[1:]
    zero_percentage = (df1[data_cols] == 0).mean()
    keep_cols = zero_percentage < threshold_value

    pos_col = df1.columns[0]
    cols_to_keep = [pos_col] + data_cols[keep_cols].tolist()

    filtered_df1 = df1[cols_to_keep].copy()

    filtered_df1 = _apply_trans(filtered_df1, trans1)
    return filtered_df1


def find_best_fold(saved_models_dir, criterion="inner_val_loss"):
    fold_dirs = glob.glob(os.path.join(saved_models_dir, "fold_*"))
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {saved_models_dir}")

    best_fold = None
    best_score = float("inf")

    for fold_dir in fold_dirs:
        config_path = os.path.join(fold_dir, "scvi_mse_config.json")
        if not os.path.exists(config_path):
            print(f"Warning: {config_path} not found, skipping")
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

    # 1) Find best model
    final_model_path = os.path.join(args.saved_models_dir, "final_model.pt")
    final_config_path = os.path.join(args.saved_models_dir, "final_config.json")

    if os.path.exists(final_model_path) and os.path.exists(final_config_path):
        print("[INFO] Using final_model (trained on all data)")
        model_path = final_model_path
        config_path = final_config_path
    else:
        print("[INFO] final_model not found, using best fold")
        fold_dir = find_best_fold(args.saved_models_dir, args.criterion)
        model_path = os.path.join(fold_dir, "scvi_mse_weights.pt")
        config_path = os.path.join(fold_dir, "scvi_mse_config.json")

    # 2) Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    print(f"\n[CONFIG]")
    print(f"  threshold: {cfg['threshold']}")
    print(f"  trans1: {cfg['trans1']}")
    print(f"  architecture: n_hidden={cfg['n_hidden']}, n_latent={cfg['n_latent']}, n_layers={cfg['n_layers']}")
    print(f"  beta: {cfg['beta']}")

    # 3) Load and preprocess
    print("\n[LOAD DATA]")
    df1_raw = pd.read_feather(args.data_path1)
    print(f"  Original df1: {df1_raw.shape}")

    print("\n[PREPROCESS] Applying filter_and_transform...")
    df1_transformed = filter_and_transform(
        df1_raw, cfg['threshold'], cfg['trans1']
    )
    print(f"  Transformed df1: {df1_transformed.shape}")

    # 4) Save transformed file
    out_dir = Path(args.transformed_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path1 = out_dir / "Count_matrix_transformed_rep1.feather"
    df1_transformed.to_feather(save_path1)
    print(f"[SAVE] Transformed rep1 -> {save_path1}")

    # 5) Prepare model input
    if 'pos' in df1_transformed.columns:
        pos_col = df1_transformed['pos'].copy()
        df1_transformed = df1_transformed.drop(columns=['pos'])
    else:
        pos_col = None

    X = torch.tensor(df1_transformed.to_numpy(), dtype=torch.float32, device=device)

    input_dim = int(cfg["input_dim"])
    if X.shape[1] != input_dim:
        raise ValueError(f"Dimension mismatch! Expected {input_dim}, got {X.shape[1]}")

    # 6) Load model
    print("\n[MODEL] Loading ScVIModel...")
    model = ScVIModel(
        n_input=input_dim,
        n_hidden=cfg["n_hidden"],
        n_latent=cfg["n_latent"],
        n_layers=cfg["n_layers"],
        dropout_rate=0.1,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 7) Reconstruct
    print("[RECONSTRUCT] Running ScVIModel...")
    px_rate, q_m, logvar = model(X)

    # 8) Save reconstruction
    recon_df = pd.DataFrame(px_rate.cpu().numpy(), columns=df1_transformed.columns.tolist())
    if pos_col is not None:
        recon_df.insert(0, 'pos', pos_col.values)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recon_df.to_feather(out_path)

    print(f"\n[DONE] Reconstruction saved to: {out_path}")
    print(f"       Shape: {recon_df.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="scVI-MSE Reconstruction / Inference (single input)")
    ap.add_argument("--data_path1", required=True, help="Input feather rep1 (to reconstruct)")
    ap.add_argument("--transformed_out_dir", required=True,
                    help="Directory to save Count_matrix_transformed_rep1.feather")
    ap.add_argument("--saved_models_dir", required=True, help="saved_models directory")
    ap.add_argument("--criterion", default="inner_val_loss",
                    choices=["inner_val_loss", "outer_test_loss"])
    ap.add_argument("--out_path", required=True, help="Output reconstruction feather path")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)