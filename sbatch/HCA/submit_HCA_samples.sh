#!/bin/bash
dropout_keep_par=""
tsv="/dcs11/hongkai/data/yhu1/One_Shifting_Results/HCA_input_sample_pairs.tsv"
# run_first_n_sample=4
export folder_name="HCA"

worker_script="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/HCA/run_HCA_one_sample_array.sh"
plot_script="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/HCA/run_HCA_one_sample_plot.sh"
baseline_worker_script="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/run_scVI_DCA_Saver_multi.sh"

# train, plot, baseline
tasks="baseline"

has_task () { [[ ",${tasks// /}," == *",$1,"* ]]; }

log_dir="/dcs11/hongkai/data/yhu1/One_Shifting_Results/HCA/AAA_logs"
mkdir -p "$log_dir"

# tail -n +2 "$tsv" | awk -v n="$run_first_n_sample" 'NR <= n {print $2}' | while read -r sample_name; do
selected_rows="1,2"
# selected_rows="8,9"
# selected_rows="10,11,12,13"
# selected_rows="14,15,16,17,18,19"
# selected_rows="8,9,10,11,12,13,14,15,16,17,18,19"

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

    train_job_id=""

    if has_task train; then
        echo "Submitting training array for sample_name=${sample_name}"

        train_job_id=$(sbatch \
            --job-name="HCA_${dropout_keep_par}_${sample_name}" \
            --exclude=compute-170 \
            --export=ALL,folder_name="${folder_name}",sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
            --output="${log_dir}/submit_${folder_name}_gpu_${dropout_keep_par}_${sample_name}_%a.out" \
            --error="${log_dir}/submit_${folder_name}_gpu_${dropout_keep_par}_${sample_name}_%a.err" \
            "$worker_script" | awk '{print $4}')

        echo "Submitted training array job: ${train_job_id}"
    fi

    if has_task plot; then
        echo "Submitting dependent plot job for sample_name=${sample_name}"

        # only chain the dependency when this sample's train job was actually submitted
        if [[ -n "$train_job_id" ]]; then
            plot_dependency=(--dependency=afterok:${train_job_id})
        else
            plot_dependency=()
            echo "  (no train job in this run, submitting the plot job without a dependency)"
        fi

        plot_job_id=$(sbatch \
            --job-name="HCA_plot_${dropout_keep_par}_${sample_name}" \
            "${plot_dependency[@]}" \
            --exclude=compute-170 \
            --export=ALL,sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
            --output="${log_dir}/plot_HCA_${dropout_keep_par}_${sample_name}_%A.out" \
            --error="${log_dir}/plot_HCA_${dropout_keep_par}_${sample_name}_%A.err" \
            "$plot_script" | awk '{print $4}')

        echo "Submitted plot array job: ${plot_job_id}, dependency=${train_job_id:-none}"
    fi

    # scVI, DCA, Saver
    if has_task baseline; then
        echo "Submitting training baseline array for sample_name=${sample_name}"

        baseline_job_id=$(sbatch \
            --job-name="HCA_scVI_DCA_Saver_${sample_name}" \
            --exclude=compute-170 \
            --export=ALL,folder_name="${folder_name}",sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
            --output="${log_dir}/submit_${folder_name}_baseline_${dropout_keep_par}_${sample_name}_%a.out" \
            --error="${log_dir}/submit_${folder_name}_baseline_${dropout_keep_par}_${sample_name}_%a.err" \
            "$baseline_worker_script" | awk '{print $4}')

        echo "Submitted training baseline array job: ${baseline_job_id}"
    fi
done
