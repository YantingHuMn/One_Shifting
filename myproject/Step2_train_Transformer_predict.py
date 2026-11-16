"""
Train a Transformer for one of the MLP prediction tasks (Array 0-5), in place of the MLP.

--task within      Array 0/1, TARGET_SPACE=transformed (target follows input trans/norm)
--task within_raw  Array 0/1, TARGET_SPACE=raw_counts  (target fixed no_norm + no_trans)
--task cross       Array 2-5                            (target fixed no_norm + no_trans)

Everything except the network is mlp_common.py, so the results compare directly with the
MLP runs. --hidden_grid1/2 are not used by the Transformer.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_common
import transformer_predict_common as tpc

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train a multi-output Transformer: input genes -> target genes")
    mlp_common.add_train_args(p)
    tpc.add_transformer_args(p)
    args = p.parse_args()
    if args.task in ("within_raw", "cross") and (
            args.target_trans != "no_trans" or args.target_norm != "no_norm"):
        p.error(f"--task {args.task} requires --target_trans no_trans --target_norm no_norm")
    args.hidden_grid1 = args.hidden_grid2 = "0"     # MLP-only; recorded as 0
    tpc.use_transformer(args)
    mlp_common.run_training(args, task=args.task)