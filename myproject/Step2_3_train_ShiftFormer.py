import argparse
import json
import os
import sys

import shiftformer

parser = argparse.ArgumentParser(description='ShiftFormer: Transformer_denoise transformation x normalization selection')
parser.add_argument('--train', type=str, required=True, help='Raw count matrix feather (no dropout/norm/trans)')
parser.add_argument('--test', type=str, required=True, help='Dropout count matrix feather (same cells/genes)')
parser.add_argument('--out_dir', type=str, required=True, help='Output directory (summary.tsv, report.txt, run_log.json)')
parser.add_argument('--pipeline_genes', type=str, default=None, help='Matrix whose genes define the original pipeline gene set')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

try:
    result = shiftformer.choose_combination(
        train=args.train,
        test=args.test,
        seed=args.seed,
        pipeline_genes=args.pipeline_genes,
        threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
except shiftformer.StopForUser as e:
    with open(os.path.join(args.out_dir, "run_log.json"), "w") as f:
        json.dump(e.log, f, indent=2, default=str)
    print(f"[STOP] {e}; details in {args.out_dir}/run_log.json", file=sys.stderr)
    sys.exit(2)

result.save(args.out_dir)
print(result.report)

if result.final:
    sys.exit(0)

if result.n_successful > 0:
    print(
        "[WARNING] Some combinations failed; saved results are provisional.",
        file=sys.stderr
    )
    sys.exit(3)

print("[FATAL] No combinations completed successfully.", file=sys.stderr)
sys.exit(1)
