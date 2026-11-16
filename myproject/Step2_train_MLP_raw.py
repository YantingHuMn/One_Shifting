"""
Task 0/1 with a raw-count target: TF features -> non-TF HVGs of the same modality, where
the target is always the no_norm + no_trans matrix. Only the input sweeps trans x norm.
The implementation is shared and lives in mlp_common.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_common

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train a multi-output MLP: TF features -> non-TF HVGs (raw-count target)")
    mlp_common.add_train_args(p)
    args = p.parse_args()
    if args.target_trans != "no_trans" or args.target_norm != "no_norm":
        p.error("raw-count target requires --target_trans no_trans --target_norm no_norm")
    mlp_common.run_training(args, task="within_raw")