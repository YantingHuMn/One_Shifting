#!/bin/bash
#SBATCH --job-name=MLP
#SBATCH --partition=shared
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL

cd /dcs10/hongkai/data/yhu1/One_Shifting

if [[ -z "${sample_name}" ]]; then
    echo "ERROR: sample_name is not set"; exit 1
fi
if [[ -z "${folder_name}" ]]; then
    echo "ERROR: folder_name is not set"; exit 1
fi

export sample_name folder_name
export dropout_keep_par=""

task_id=${SLURM_ARRAY_TASK_ID:-0}
case "$task_id" in
    0) mlp_kind="within"; export mlp_modality="RNA"  ;;
    1) mlp_kind="within"; export mlp_modality="ATAC" ;;
    2) mlp_kind="cross";  export mlp_modality=""     ;;
    *) echo "ERROR: SLURM_ARRAY_TASK_ID=${task_id} is not one of 0,1,2"; exit 1 ;;
esac

echo "======================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  ARRAY_TASK_ID=${task_id}"
echo "sample_name=${sample_name}  folder_name=${folder_name}"
echo "mlp_kind=${mlp_kind}  mlp_modality=${mlp_modality}"
echo "HOSTNAME=$(hostname)"
echo "======================================"

if [[ "$mlp_kind" == "within" ]]; then
    bash ../One_Shifting/inst/scripts/run_MLP_within.sh \
        ../One_Shifting/inst/config/config_run_MLP_within_multi.sh
else
    bash ../One_Shifting/inst/scripts/run_MLP_cross.sh \
        ../One_Shifting/inst/config/config_run_MLP_cross_multi.sh
fi