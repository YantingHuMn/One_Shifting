#!/bin/bash
#SBATCH --job-name=check_count_matrix
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --output=/dcs07/hongkai/data/yhu1/One_Shifting_Results/slurm_check_count_matrix_%A_%a.out
#SBATCH --error=/dcs07/hongkai/data/yhu1/One_Shifting_Results/slurm_check_count_matrix_%A_%a.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL

folder_names=(
    "HCA"
    "ENCODE"
    "10x"
)

folder_name="${folder_names[$SLURM_ARRAY_TASK_ID]}"

selected_rows=""

base_dir="/dcs07/hongkai/data/yhu1/One_Shifting_Results"

tsv="${base_dir}/${folder_name}_input_sample_pairs.tsv"
read_dir="${base_dir}/${folder_name}"
output_csv="${read_dir}/${folder_name}_check_count_matrix_na_row.csv"

log_dir="${read_dir}/AAA_logs"
mkdir -p "$log_dir"

log_file="${log_dir}/AAA_${folder_name}_check_matrix_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$log_file") 2>&1

module load conda_R

echo "======================================"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Folder name: ${folder_name}"
echo "TSV file: ${tsv}"
echo "Read directory: ${read_dir}"
echo "Output CSV: ${output_csv}"
echo "Log file: ${log_file}"
echo "Start time: $(date)"
echo "======================================"

if [[ ! -f "$tsv" ]]; then
    echo "ERROR: TSV file not found: $tsv"
    exit 1
fi

if [[ ! -d "$read_dir" ]]; then
    echo "ERROR: Read directory not found: $read_dir"
    exit 1
fi

# Each array task has a different output CSV, so there is no cross-task conflict.
rm -f "$output_csv"

tail -n +2 "$tsv" |
awk -v rows="$selected_rows" '
BEGIN {
    if (rows == "") {
        all = 1
    } else {
        split(rows, selected, ",")
        for (i in selected) {
            keep[selected[i]] = 1
        }
    }
}
all || keep[NR] {
    print $2
}
' |
while read -r sample_name; do

    if [[ -z "$sample_name" ]]; then
        echo "Skipping empty sample_name"
        continue
    fi

    echo
    echo "======================================"
    echo "Processing sample_name=${sample_name}"
    echo "Folder=${folder_name}"
    echo "Start sample: $(date)"
    echo "======================================"

    Rscript \
        /dcs10/hongkai/data/yhu1/One_Shifting/R/check_count_matrix_na_row.R \
        "$sample_name" \
        "$read_dir" \
        "$output_csv"

    exit_status=$?

    if [[ $exit_status -ne 0 ]]; then
        echo "ERROR: sample ${sample_name} failed with status ${exit_status}"
    else
        echo "Completed sample ${sample_name}: $(date)"
    fi

done

echo
echo "--------------------------------------"
echo "Folder finished: ${folder_name}"
echo "Output saved to: ${output_csv}"
echo "Finish time: $(date)"
echo "--------------------------------------"