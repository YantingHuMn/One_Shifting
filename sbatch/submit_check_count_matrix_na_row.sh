#!/bin/bash
#SBATCH --job-name=check_count_matrix
#SBATCH --partition=shared
#SBATCH --time=12:00:00
#SBATCH --array=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/dcs07/hongkai/data/yhu1/One_Shifting_Results/logs/slurm_check_count_matrix_%A_%a.out
#SBATCH --error=/dcs07/hongkai/data/yhu1/One_Shifting_Results/logs/slurm_check_count_matrix_%A_%a.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL

cd /dcs10/hongkai/data/yhu1/One_Shifting

folder_names=(
    # "HCA"
    "ENCODE"
    # "10x"
)

folder_name="${folder_names[$SLURM_ARRAY_TASK_ID]}"

selected_rows=""
dropout_keep_par=""
base_dir="/dcs07/hongkai/data/yhu1/One_Shifting_Results"

tsv="${base_dir}/${folder_name}_input_sample_pairs.tsv"
read_dir="${base_dir}/${folder_name}"
output_clear_csv="${read_dir}/${folder_name}_check_count_matrix_clear_row.csv"
output_na_csv="${read_dir}/${folder_name}_check_count_matrix_na_row.csv"

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
rm -f "$output_clear_csv"
rm -f "$output_na_csv"
rm -f ${read_dir}/${folder_name}_check_method_complete.csv
rm -f ${read_dir}/${folder_name}_data_description.csv

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

    Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/check_clear_method_summary.R \
        "$sample_name" \
        "$read_dir" \
        "$output_clear_csv"

    Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/check_count_matrix_na_row.R \
        "$sample_name" \
        "$read_dir" \
        "$output_na_csv"

    Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/check_method_complete.R \
        "$sample_name" \
        "$read_dir" \
        "${read_dir}/${folder_name}_check_method_complete.csv"

    Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/record_data_description.R \
        "$sample_name" \
        "$read_dir" \
        "${read_dir}/${folder_name}_data_description.csv"

    export sample_name
    export folder_name="${folder_name}"
    export dropout_keep_par

    bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
        "../One_Shifting/inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh"

    bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
        "../One_Shifting/inst/config/config_run_multi_ATAC_HCA_10x_ENCODE.sh"

    echo "Finished sample_name=${sample_name}"
    echo "End sample: $(date)"

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