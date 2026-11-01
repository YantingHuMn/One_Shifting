"""
Task 2 prediction: load the single saved cross-modality model and predict the test cells.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_common

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Predict test cells with the saved cross-modality MLP")
    mlp_common.add_predict_args(p, default_label="mlp_atac_to_rna")
    mlp_common.run_prediction(p.parse_args(), task="cross")