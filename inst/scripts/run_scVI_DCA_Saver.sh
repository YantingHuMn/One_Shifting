
#!/bin/bash
CONFIG_FILE="$1"
LOSS="$2"
STEPS="${3:-scVI,DCA,SAVER,Figure}"

if [ -z "$CONFIG_FILE" ] || [ -z "$LOSS" ]; then
    echo "Usage: bash run_pipeline.sh <config_file> <LOSS> [STEPS]"
    echo "  config_file: path to config (e.g. configs/pbmc_atac_rna.sh)"
    echo "  LOSS: zinb or other method"
    echo "  STEPS: scVI,DCA,SAVER,Figure (default: all)"
    exit 1
fi

# Validate requested steps
IFS=',' read -ra SELECTED_STEPS <<< "$STEPS"
for step in "${SELECTED_STEPS[@]}"; do
    case "$step" in
        scVI|DCA|SAVER|Figure) ;;
        *)
            echo "Error: Unknown step '$step'"
            echo "Valid steps: scVI,DCA,SAVER,Figure"
            exit 1
            ;;
    esac
done

run_step() {
    [[ ",${STEPS}," == *",$1,"* ]]
}

source "$CONFIG_FILE"

echo "INPUT_FILE: $INPUT_FILE"
echo "GROUND_TRUTH_FILE: $GROUND_TRUTH_FILE"
echo "READ_DIR: $READ_DIR"
echo "INPUT_CATEGORY: $V1"
echo "OUTPUT_CATEGORY: $V2"
echo "V2_norm_factor: $V2_norm_factor"
echo "V2_trans_factor: $V2_trans_factor"
echo "dropout_keep_par: $dropout_keep_par"
echo "STEPS: $STEPS"

CONDITION="given_${V2}_${V2_norm_factor}_${V2_trans_factor}"
INPUT_PATH="${READ_DIR}/${V1}/Count_Matrix_norm_by_no_norm.feather"
GROUND_TRUTH_PATH="${READ_DIR}/${V2}/Count_Matrix_norm_by_no_norm.feather"

methods=("scVI" "DCA" "SAVER")
for method in "${methods[@]}"; do
    mkdir -p "${READ_DIR}/${method}"
done

export NUMBA_CACHE_DIR="${READ_DIR}/.numba_cache"
export MPLCONFIGDIR="${READ_DIR}/.mpl_cache"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

LOCK_FILE="${READ_DIR}/.count_matrix_done"
if [ ! -f "$LOCK_FILE" ]; then
    echo "Waiting for count matrix to be built by another job..."
    while [ ! -f "$LOCK_FILE" ]; do
        sleep 10
    done
fi
echo "Count matrix ready, proceeding..."


if run_step scVI; then
    echo "======Run scVI======"
    source ~/.bashrc
    conda activate scvi_env
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

    python ../One_Shifting/myproject/Step2_3_train_scVI.py \
        --input "${INPUT_PATH}" \
        --output "${READ_DIR}/scVI/reconstruct_scVI_trans_by_no_trans_norm_by_no_norm.feather" \
        --loss "$LOSS" || { echo "[FATAL] scVI failed" >&2; exit 1; }

    python -c "import torch; torch.cuda.empty_cache(); del torch; print('GPU cleared')"
    conda deactivate
fi


if run_step DCA; then
    echo "======Run DCA======"
    source ~/.bashrc
    conda activate dca_env
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

    # use default nb-conddisp for dca
    python ../One_Shifting/myproject/Step2_3_train_DCA.py \
        --input "$INPUT_PATH" \
        --output "${READ_DIR}/DCA/reconstruct_DCA_trans_by_no_trans_norm_by_no_norm.feather" \
        --ae_type "nb-conddisp" || { echo "[FATAL] DCA failed" >&2; exit 1; }

    python -c "import torch; torch.cuda.empty_cache(); del torch; print('GPU cleared')"
    conda deactivate
fi


if run_step SAVER; then
    echo "======Run SAVER======"
    module load conda_R

    Rscript ../One_Shifting/myproject/Step2_3_train_Saver.R \
        "$INPUT_PATH" \
        "${READ_DIR}/SAVER/reconstruct_SAVER_trans_by_no_trans_norm_by_no_norm.feather" \
        "${SLURM_CPUS_PER_TASK:-1}" || { echo "[FATAL] SAVER failed" >&2; exit 1; }
fi


if run_step Figure; then
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
                "$Figure_DIR" \
                "${READ_DIR}/${method}" \
                "$corr_dir" \
                "${V1}" \
                "${method}"

            sleep 2
        done
    done
fi
