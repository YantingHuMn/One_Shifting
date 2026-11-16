"""
Predict the test cells with a Transformer trained by Step2_train_Transformer_predict.py.

Same evaluation and outputs as the MLP entry points (Step3_predict_MLP{,_raw,_cross}.py):
--task within reverses the target trans/norm to raw counts, --task within_raw and cross
compare to the fixed raw target directly. The architecture is read back from
final_config.json; a model of another architecture is refused.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_common
import transformer_predict_common as tpc

LABELS = {"within": "transformer_tf_to_hvg", "within_raw": "transformer_tf_to_hvg_raw",
          "cross": "transformer_cross"}

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Predict test cells with the saved Transformer")
    mlp_common.add_predict_args(p, default_label=None)
    p.add_argument("--task", required=True, choices=sorted(LABELS))
    args = p.parse_args()
    if args.method_label is None:
        args.method_label = LABELS[args.task]
    tpc.use_transformer()
    mlp_common.run_prediction(args, task=args.task)