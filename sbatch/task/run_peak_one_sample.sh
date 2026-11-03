#!/bin/bash
#SBATCH --job-name=peak
#SBATCH --partition=shared
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL

cd /dcs10/hongkai/data/yhu1/One_Shifting

if [[ -z "${sample_name}" ]]; then
    echo "ERROR: sample_name is not set"
    exit 1
fi

dropout_keep_par="${dropout_keep_par:-}"
read -r -a methods <<< "${methods_par:-}"

if (( ${#methods[@]} == 0 )); then
    echo "ERROR: methods_par is empty"
    exit 1
fi

export sample_name
export folder_name
export dropout_keep_par

configs=(
    "../One_Shifting/inst/config/config_run_peak_mse.sh"
)

n_methods=${#methods[@]}
n_configs=${#configs[@]}
n_total=$((n_methods * n_configs))

task_id=${SLURM_ARRAY_TASK_ID}

if (( task_id >= n_total )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${task_id} exceeds total tasks=${n_total}"
    exit 1
fi

method_id=$((task_id % n_methods))
config_id=$((task_id / n_methods))

method=${methods[$method_id]}
config=${configs[$config_id]}


echo "======================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "sample_name=${sample_name}"
echo "folder_name=${folder_name}"
echo "method=${method}"
echo "dropout_keep_par=${dropout_keep_par}"
echo "config=${config}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HOSTNAME=$(hostname)"
echo "======================================"

nvidia-smi

bash ../One_Shifting/inst/scripts/run_peak_mse.sh \
    "$config" \
    "$method"