#!/bin/bash
READ_DIR="$1"
V1="$2"
V2="$3"
LOSS="$4"

INPUT_PATH="${READ_DIR}/${V1}/Count_Matrix_norm_by_no_norm.feather"
GROUND_TRUTH_PATH="${READ_DIR}/${V2}/Count_Matrix_norm_by_no_norm.feather"

methods=("scVI" "DCA" "SAVER")
for method in "${methods[@]}"; do
    mkdir -p ${READ_DIR}/${method}
done

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


echo "======Figures: Correlation Scatter Plots======"

module load conda_R

methods=("scVI" "DCA" "SAVER")
for method in "${methods[@]}"; do
    for corr_dir in col row; do
        Figure_DIR="${READ_DIR}/${method}/Figures_${corr_dir}"
        mkdir -p "$Figure_DIR"

        Rscript ../One_Shifting/R/run_correlation_scatter_scVI_DCA_Saver.R \
            "$GROUND_TRUTH_PATH" \
            "$INPUT_PATH" \
            "${READ_DIR}/${method}/reconstruct_${method}_trans_by_no_trans_norm_by_no_norm.feather" \
            "$Figure_DIR"  \
            "${READ_DIR}/${method}" \
            "$corr_dir" \
            "${V1}" \
            "${method}"

        sleep 2
    done
done



echo "======Run Comparison======"
module load conda_R

mse_methods=("VAE" "scVI_mse" "DCA_mse")
methods=("scVI" "DCA" "SAVER")

# mse vs non-mse
for method in "${methods[@]}"; do
    for mse_method in "${mse_methods[@]}"; do
        for corr_dir in col row; do
            x_dir="${READ_DIR}/${mse_method}/given_${V2}_no_norm_no_trans"
            y_dir="${READ_DIR}/${method}"
            
            Rscript ../One_Shifting/R/run_x_mag_compare_scatter.R \
                "${x_dir}" \
                "${y_dir}" \
                $V1 \
                $V2 \
                $corr_dir
        done
    done
done

# mse vs mse (pairwise)
for ((i=0; i<${#mse_methods[@]}; i++)); do
    for ((j=i+1; j<${#mse_methods[@]}; j++)); do
        for corr_dir in col row; do
            x_dir="${READ_DIR}/${mse_methods[$i]}/given_${V2}_no_norm_no_trans"
            y_dir="${READ_DIR}/${mse_methods[$j]}/given_${V2}_no_norm_no_trans"
            
            Rscript ../One_Shifting/R/run_x_mag_compare_scatter.R \
                "${x_dir}" \
                "${y_dir}" \
                $V1 \
                $V2 \
                $corr_dir
        done
    done
done
