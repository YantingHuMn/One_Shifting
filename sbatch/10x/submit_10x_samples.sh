#!/bin/bash
dropout_keep_par=""
tsv="/dcs07/hongkai/data/yhu1/One_Shifting_Results/10x_input_sample_pairs.tsv"
# run_first_n_sample=9

worker_script="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/10x/run_10x_one_sample_array.sh"
plot_script="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/10x/run_10x_one_sample_plot.sh"

log_dir="/dcs07/hongkai/data/yhu1/One_Shifting_Results/10x/AAA_logs"
mkdir -p "$log_dir"

# tail -n +2 "$tsv" | awk -v n="$run_first_n_sample" 'NR <= n {print $2}' | while read -r sample_name; do
selected_rows="1,2"

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

    echo "Submitting training array for sample_name=${sample_name}"

    train_job_id=$(sbatch \
        --job-name="10x_${dropout_keep_par}_${sample_name}" \
        --exclude=compute-170 \
        --export=ALL,sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
        --output="${log_dir}/submit_10x_gpu_${dropout_keep_par}_${sample_name}_%a.out" \
        --error="${log_dir}/submit_10x_gpu_${dropout_keep_par}_${sample_name}_%a.err" \
        "$worker_script" | awk '{print $4}')

    echo "Submitted training array job: ${train_job_id}"

    # echo "Submitting dependent plot job for sample_name=${sample_name}"

    # plot_job_id=$(sbatch \
    #     --job-name="10x_plot_${dropout_keep_par}_${sample_name}" \
    #     --dependency=afterok:${train_job_id} \
    #     --exclude=compute-170 \
    #     --export=ALL,sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
    #     --output="${log_dir}/plot_10x_${dropout_keep_par}_${sample_name}_%A.out" \
    #     --error="${log_dir}/plot_10x_${dropout_keep_par}_${sample_name}_%A.err" \
    #     "$plot_script" | awk '{print $4}')

    # echo "Submitted plot job: ${plot_job_id}, dependency=afterok:${train_job_id}"
    # echo "--------------------------------------"
done