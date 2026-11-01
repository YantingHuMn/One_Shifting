# Task 1 config for the multi-sample submitter. Derives every path from the environment
# the worker exports, exactly like config_run_multi_ATAC_HCA_10x_ENCODE.sh does.
#
# mlp_modality picks which modality supplies BOTH the TF input and the HVG target, so the
# same config covers RNA TF -> RNA HVG and gene-activity TF -> gene-activity HVG.
echo "DEBUG sample_name='$sample_name'  folder_name='$folder_name'  mlp_modality='$mlp_modality'"

OUT_DIR="/dcs11/hongkai/data/yhu1/One_Shifting_Results"
READ_DIR="${OUT_DIR}/${folder_name}/${sample_name}_INPUT_ATAC"
MODALITY="${sample_name}_${mlp_modality:-RNA}"

N_HVG=1000
MIN_CELLS=3
TEST_FRAC=0.2
VAL_FRAC=0.1
SEED=42

EPOCHS=100
PATIENCE=15
HIDDEN1=256
HIDDEN2=128
LR=0.0001
BATCH=64
DROPOUT=0.1