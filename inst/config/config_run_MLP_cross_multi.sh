# Task 2 config for the multi-sample submitter.
echo "DEBUG sample_name='$sample_name'  folder_name='$folder_name'"

OUT_DIR="/dcs11/hongkai/data/yhu1/One_Shifting_Results"
# mlp_in / mlp_out come from the worker (unset -> Task 2: ATAC -> RNA). Both modalities are
# read from the input modality's own <sample>_INPUT_<mlp_in> folder, so cells stay paired.
READ_DIR="${OUT_DIR}/${folder_name}/${sample_name}_INPUT_${mlp_in:-ATAC}"
V1="${sample_name}_${mlp_in:-ATAC}"
V2="${sample_name}_${mlp_out:-RNA}"
PREP_TASK="${mlp_prep:-cross}"
OUTPUT_SUFFIX="${mlp_tag:-}"
METHOD_LABEL="${mlp_label:-mlp_atac_to_rna_same_genes}"

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