"""
Task 1 -- within one modality: TF features -> non-TF highly variable genes.

One multi-output MLP predicts every target gene at once. The modality is whatever
prep_mlp_data.R was pointed at: RNA in, RNA TF -> RNA HVG; gene activity in, gene-activity
TF -> gene-activity HVG. The implementation is shared with the cross-modality task and
lives in mlp_common.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_common

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train a multi-output MLP: TF features -> non-TF HVGs (same modality)")
    mlp_common.add_train_args(p)
    mlp_common.run_training(p.parse_args(), task="within")