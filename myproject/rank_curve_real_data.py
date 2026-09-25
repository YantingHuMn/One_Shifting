#!/usr/bin/env python
"""Rank-resolved cross-modality signal on REAL data. No training, no GPU.

For each transformation, take the centred SVD of V1, keep the top-k components,
and measure the mean per-gene Pearson r between the rank-k reconstruction and V2.
This is 'what a model with exactly k latent dimensions could achieve at best',
so it separates the geometry of the data from anything about training.

Usage:
  python rank_curve_real_data.py \
      --data_path1 <READ_DIR>/<V1>/Count_Matrix_norm_by_<S>.feather \
      --data_path2 <READ_DIR>/<V2>/Count_Matrix_norm_by_no_norm.feather \
      --out rank_curve_<V1>_norm<S>.csv

Run it once per norm factor S you care about; the curve shape depends on S.
"""
import argparse
import numpy as np
import pandas as pd


TRANS = {
    "no_trans":      lambda x: x,
    "count+1":       lambda x: x + 1,
    "sqrt":          lambda x: np.sqrt(x),
    "sqrt+1":        lambda x: np.sqrt(x + 1),
    "log2":          lambda x: np.log2(x + 1),
    "log2(count+2)": lambda x: np.log2(x + 2),
}
PAIRS = [("no_trans", "count+1"), ("sqrt", "sqrt+1"), ("log2", "log2(count+2)")]


def load_pair(p1, p2, threshold=1.0):
    df1, df2 = pd.read_feather(p1), pd.read_feather(p2)
    for df in (df1, df2):
        for c in ("pos", "barcode"):
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    cols = [c for c in df1.columns if c in set(df2.columns)]
    df1, df2 = df1[cols], df2[cols]
    keep = (df1 == 0).mean() < threshold
    df1, df2 = df1.loc[:, keep], df2.loc[:, keep]
    A = df1.to_numpy(dtype=np.float64)
    B = df2.to_numpy(dtype=np.float64)
    ok = (A.std(0) > 0) & (B.std(0) > 0) & np.isfinite(A).all(0) & np.isfinite(B).all(0)
    return A[:, ok], B[:, ok]


def rank_curve(Xt, V2, ks):
    Xc = Xt - Xt.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    V2c = V2 - V2.mean(0, keepdims=True)
    v2n = V2c / (np.linalg.norm(V2c, axis=0) + 1e-12)
    out = []
    for k in ks:
        R = (U[:, :k] * s[:k]) @ Vt[:k]
        Rn = R / (np.linalg.norm(R, axis=0) + 1e-12)
        out.append(float(np.mean((Rn * v2n).sum(0))))
    ev = s ** 2 / (s ** 2).sum()
    return out, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path1", required=True, help="V1 (input) feather, cell x gene")
    ap.add_argument("--data_path2", required=True, help="V2 (ground truth) feather, cell x gene")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--out", default="rank_curve.csv")
    a = ap.parse_args()

    V1, V2 = load_pair(a.data_path1, a.data_path2, a.threshold)
    print(f"[data] {V1.shape[0]} cells x {V1.shape[1]} genes, "
          f"V1 zero fraction {float((V1 == 0).mean()):.3f}", flush=True)

    kmax = min(V1.shape) - 1
    ks = [k for k in (1, 2, 3, 5, 8, 10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
          if k <= kmax]
    rows, spec = [], []
    for name, fn in TRANS.items():
        Xt = fn(V1)
        curve, ev = rank_curve(Xt, V2, ks)
        for k, v in zip(ks, curve):
            rows.append(dict(trans=name, k=k, mean_r=v))
        spec.append(dict(trans=name, var_top10=float(ev[:10].sum()),
                         var_top32=float(ev[:32].sum()),
                         n_pc_90=int(np.searchsorted(np.cumsum(ev), 0.90) + 1)))
        print(f"[{name:14s}] r@10={curve[ks.index(10)]:+.4f} "
              f"r@32={curve[ks.index(32)]:+.4f} r@max={curve[-1]:+.4f} "
              f"n_pc_90={spec[-1]['n_pc_90']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.DataFrame(spec).to_csv(a.out.replace(".csv", "_spectrum.csv"), index=False)

    p = df.pivot_table(index="k", columns="trans", values="mean_r")
    d = pd.DataFrame({f"{sh} - {ba}": p[sh] - p[ba] for ba, sh in PAIRS})
    d.to_csv(a.out.replace(".csv", "_diff.csv"))
    print("\n=== shifted - original, by rank budget k ===")
    print("(negative = one-shifting HURTS a model with only k latent dims)")
    print(d.round(4).to_string())
    print(f"\n[saved] {a.out}, *_spectrum.csv, *_diff.csv")


if __name__ == "__main__":
    main()