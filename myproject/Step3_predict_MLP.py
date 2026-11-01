"""
Task 1 prediction: load the single saved multi-output model and predict the test cells.

Input columns are re-ordered to the gene order recorded at training time and test cell
ids are checked against the recorded train/validation cells, so a reshuffled or
mis-specified matrix raises instead of producing plausible-looking numbers.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_common

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Predict test cells with the saved within-modality MLP")
    mlp_common.add_predict_args(p, default_label="mlp_tf_to_hvg")
    mlp_common.run_prediction(p.parse_args(), task="within")