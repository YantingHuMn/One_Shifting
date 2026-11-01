"""
Task 2 -- cross modality: ATAC gene-activity scores -> RNA expression of the same cells.

Input defaults to every gene-activity feature that survived the training-cell filter, not
only TFs; the target defaults to the non-TF RNA HVGs. Both gene sets come from TXT lists
written by prep_mlp_data.R, so the two sets need not match in content or in size and the
cell pairing is by identifier. Pointing --input_train/--target_train at Task 1's output
directory reuses its split and its RNA target list for a matched comparison.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_common

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train a multi-output MLP: ATAC gene activity -> RNA HVGs")
    mlp_common.add_train_args(p)
    mlp_common.run_training(p.parse_args(), task="cross")