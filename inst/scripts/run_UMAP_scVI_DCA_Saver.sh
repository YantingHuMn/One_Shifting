#!/bin/bash
READ_DIR="$1"
V1="$2"
LOSS="$3"

INPUT_PATH="${READ_DIR}/${V1}/Count_Matrix_norm_by_no_norm.feather"

methods=("scVI" "DCA" "SAVER")
for method in "${methods[@]}"; do
    mkdir -p ${READ_DIR}/${method}
done

LOCK_FILE="${READ_DIR}/${V1}/.count_matrix_done"
if [ ! -f "$LOCK_FILE" ]; then
    echo "Waiting for count matrix to be built by another job..."
    while [ ! -f "$LOCK_FILE" ]; do
        sleep 10
    done
fi
echo "Count matrix ready, proceeding..."

echo "======Run scVI======"
source ~/.bashrc
conda activate scvi_env
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

python ../One_Shifting/myproject/Step2_3_train_scVI.py \
    --input "${INPUT_PATH}" \
    --output "${READ_DIR}/scVI/reconstruct_scVI_trans_by_no_trans_norm_by_no_norm.feather" \
    --loss "$LOSS" 

python -c "import torch; torch.cuda.empty_cache(); del torch; print('GPU cleared')"
conda deactivate 


echo "======Run DCA======"
source ~/.bashrc
conda activate dca_env
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

python ../One_Shifting/myproject/Step2_3_train_DCA.py \
    --input "$INPUT_PATH" \
    --output "${READ_DIR}/DCA/reconstruct_DCA_trans_by_no_trans_norm_by_no_norm.feather" \
    --ae_type "${LOSS}"

python -c "import torch; torch.cuda.empty_cache(); del torch; print('GPU cleared')"
conda deactivate 


echo "======Run SAVER======"
module load conda_R

Rscript ../One_Shifting/myproject/Step2_3_train_Saver.R \
    "$INPUT_PATH" \
    "${READ_DIR}/SAVER/reconstruct_SAVER_trans_by_no_trans_norm_by_no_norm.feather" \
    $(nproc)

