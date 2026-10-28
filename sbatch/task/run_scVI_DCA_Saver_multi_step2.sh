#!/bin/bash
#SBATCH --job-name=multi_scVI_DCA_Saver
#SBATCH --partition=shared     
# #SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=12
#SBATCH --output=/dcs10/hongkai/data/yhu1/One_Shifting_Results/logs/submit_scVI_DCA_Saver_%a.out
#SBATCH --error=/dcs10/hongkai/data/yhu1/One_Shifting_Results/logs/submit_scVI_DCA_Saver_%a.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-1

cd /dcs10/hongkai/data/yhu1/One_Shifting

if [[ -z "${sample_name}" ]]; then
    echo "ERROR: sample_name is not set"
    exit 1
fi

dropout_keep_par="${dropout_keep_par:-}"
LOSS="zinb"

export sample_name
export folder_name
export dropout_keep_par

echo $LOSS

configs=(
    "../One_Shifting/inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh"
    "../One_Shifting/inst/config/config_run_multi_ATAC_HCA_10x_ENCODE.sh"
)

n_configs=${#configs[@]}
n_total=$n_configs
task_id=${SLURM_ARRAY_TASK_ID}

if (( task_id >= n_total )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${task_id} exceeds total tasks=${n_total}"
    exit 1
fi

config=${configs[$task_id]}

echo "Array task $SLURM_ARRAY_TASK_ID: READ_DIR=$READ_DIR, V1=$V1, V2=$V2"

bash ../One_Shifting/inst/scripts/run_scVI_DCA_Saver.sh \
    "$config" \
    $LOSS \
    "SAVER,Figure"
