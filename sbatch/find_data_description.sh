#!/bin/bash
folder_name="HCA"

dropout_keep_par=""
tsv="/dcs07/hongkai/data/yhu1/One_Shifting_Results/${folder_name}_input_sample_pairs.tsv"

worker_script="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/${folder_name}/run_${folder_name}_one_sample_array.sh"
plot_script="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/${folder_name}/run_${folder_name}_one_sample_plot.sh"

log_dir="/dcs07/hongkai/data/yhu1/One_Shifting_Results/${folder_name}/AAA_logs"
mkdir -p "$log_dir"

log_file="${log_dir}/AAA_${folder_name}_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1

selected_rows=""

read_dir="/dcs07/hongkai/data/yhu1/One_Shifting_Results/${folder_name}"

rm ${read_dir}/${folder_name}_check_method_complete.csv
rm ${read_dir}/${folder_name}_data_description.csv

module load conda_R

echo "Log file: $log_file"
echo "Start time: $(date)"
echo "--------------------------------------"

tail -n +2 "$tsv" | awk -v rows="$selected_rows" '
BEGIN {
    if (rows == "") {
        all = 1
    } else {
        split(rows, a, ",")
        for (i in a) keep[a[i]] = 1
    }
}
all || keep[NR] {print $2}
' | while read -r sample_name; do

    if [[ -z "$sample_name" ]]; then
        echo "Skipping empty sample_name"
        continue
    fi

    echo "======================================"
    echo "Processing sample_name=${sample_name}"
    echo "Start sample: $(date)"
    echo "======================================"

    cd /dcs10/hongkai/data/yhu1/One_Shifting || exit 1

    Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/check_method_complete.R \
        "$sample_name" \
        "$read_dir" \
        "${read_dir}/${folder_name}_check_method_complete.csv"

    export sample_name
    export folder_name="${folder_name}"
    export dropout_keep_par

    bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
        "../One_Shifting/inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh"

    bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
        "../One_Shifting/inst/config/config_run_multi_ATAC_HCA_10x_ENCODE.sh"

    Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/record_data_description.R \
        "$sample_name" \
        "$read_dir" \
        "${read_dir}/${folder_name}_data_description.csv"

    echo "Finished sample_name=${sample_name}"
    echo "End sample: $(date)"

done

echo "--------------------------------------"
echo "All finished: $(date)"