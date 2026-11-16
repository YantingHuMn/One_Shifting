"""
Task 0/1 prediction with a raw-count target: the model output already is on the raw-count
scale and is compared to the raw target directly, with no reverse transformation.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_common

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Predict test cells with the saved raw-target within-modality MLP")
    mlp_common.add_predict_args(p, default_label="mlp_tf_to_hvg_raw")
    mlp_common.run_prediction(p.parse_args(), task="within_raw")