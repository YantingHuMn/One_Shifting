# Task 2 config for the multi-sample submitter.
echo "DEBUG sample_name='$sample_name'  folder_name='$folder_name'"

OUT_DIR="/dcs11/hongkai/data/yhu1/One_Shifting_Results"
READ_DIR="${OUT_DIR}/${folder_name}/${sample_name}_INPUT_ATAC"
V1="${sample_name}_ATAC"
V2="${sample_name}_RNA"

# Held fixed while the input transformation sweeps, so the sweep measures the effect on
# the input representation alone.
TARGET_NORM="no_norm"
TARGET_TRANS="no_trans"

# REUSE_LISTS_DIR="${OUT_DIR}/${folder_name}/${sample_name}_INPUT_RNA/MLP_within_${sample_name}_RNA/gene_lists"
REUSE_LISTS_DIR=""

N_HVG=1000
MIN_CELLS=3
TEST_FRAC=0.2
VAL_FRAC=0.1
SEED=42

EPOCHS=100
PATIENCE=15
HIDDEN1=512
HIDDEN2=128
LR=0.0001
BATCH=64