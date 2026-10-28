#!/bin/bash
dropout_keep_par=""
# run_first_n_sample=4
export folder_name="ENCODE"
tsv="/dcs11/hongkai/data/yhu1/One_Shifting_Results/${folder_name}_input_sample_pairs.tsv"

worker_script="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/${folder_name}/run_${folder_name}_one_sample_array.sh"
plot_script="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/${folder_name}/run_${folder_name}_one_sample_plot.sh"
baseline_worker_script_step1_gpu="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/run_scVI_DCA_Saver_multi_step1.sh"
baseline_worker_script_step2_cpu="/dcs10/hongkai/data/yhu1/One_Shifting/sbatch/run_scVI_DCA_Saver_multi_step2.sh"

# train, plot, baseline step2_only
tasks="train"

has_task () { [[ ",${tasks// /}," == *",$1,"* ]]; }

log_dir="/dcs11/hongkai/data/yhu1/One_Shifting_Results/${folder_name}/AAA_logs"
mkdir -p "$log_dir"
job_id_file="${log_dir}/AAA_step2_job_map.txt"
touch "$job_id_file"

# tail -n +2 "$tsv" | awk -v n="$run_first_n_sample" 'NR <= n {print $2}' | while read -r sample_name; do
# selected_rows="4,5,6,7,8"
# selected_rows="9,10,11"
# selected_rows=""
selected_rows="6"

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
            --job-name="${folder_name}_${dropout_keep_par}_${sample_name}" \
            --exclude=compute-170 \
            --export=ALL,folder_name="${folder_name}",sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
            --output="${log_dir}/submit_${folder_name}_gpu_${dropout_keep_par}_${sample_name}_2.out" \
            --error="${log_dir}/submit_${folder_name}_gpu_${dropout_keep_par}_${sample_name}_2.err" \
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
            --job-name="${folder_name}_plot_${dropout_keep_par}_${sample_name}" \
            "${plot_dependency[@]}" \
            --exclude=compute-170 \
            --export=ALL,sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
            --output="${log_dir}/plot_${folder_name}_${dropout_keep_par}_${sample_name}_%A.out" \
            --error="${log_dir}/plot_${folder_name}_${dropout_keep_par}_${sample_name}_%A.err" \
            "$plot_script" | awk '{print $4}')

        echo "Submitted plot array job: ${plot_job_id}, dependency=${train_job_id:-none}"
    fi

    # scVI, DCA
    if has_task baseline; then
        echo "Submitting training baseline array for sample_name=${sample_name}"

        baseline_gpu_job_id=$(sbatch \
            --job-name="${folder_name}_${sample_name}" \
            --exclude=compute-170 \
            --export=ALL,folder_name="${folder_name}",sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
            --output="${log_dir}/submit_${folder_name}_baseline_gpu_job_${dropout_keep_par}_${sample_name}_%a.out" \
            --error="${log_dir}/submit_${folder_name}_baseline_gpu_job_${dropout_keep_par}_${sample_name}_%a.err" \
            "$baseline_worker_script_step1_gpu" | awk '{print $4}')

        echo "Submitted training baseline array job: ${baseline_gpu_job_id}"
    fi

    # SAVER, Figure
    if has_task baseline; then
        echo "Submitting training baseline array for sample_name=${sample_name}"

        baseline_cpu_job_id=$(sbatch \
            --job-name="${folder_name}_${sample_name}" \
            --exclude=compute-170 \
            --dependency=aftercorr:${baseline_gpu_job_id} \
            --export=ALL,folder_name="${folder_name}",sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
            --output="${log_dir}/submit_${folder_name}_baseline_cpu_job_${dropout_keep_par}_${sample_name}_%a.out" \
            --error="${log_dir}/submit_${folder_name}_baseline_cpu_job_${dropout_keep_par}_${sample_name}_%a.err" \
            "$baseline_worker_script_step2_cpu" | awk '{print $4}')

        echo "Submitted training baseline array job: ${baseline_cpu_job_id}"
        if [[ -n "$baseline_cpu_job_id" ]]; then
            echo "$baseline_cpu_job_id" >> "$job_id_file"
        fi
    fi

    # Step2 only: SAVER, Figure
    if has_task step2_only; then
        echo "Submitting training baseline array for sample_name=${sample_name}"

        baseline_cpu_job_id=$(sbatch \
            --job-name="${folder_name}_${sample_name}" \
            --exclude=compute-170 \
            --export=ALL,folder_name="${folder_name}",sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}" \
            --output="${log_dir}/submit_${folder_name}_baseline_cpu_job_${dropout_keep_par}_${sample_name}_rerun_%a.out" \
            --error="${log_dir}/submit_${folder_name}_baseline_cpu_job_${dropout_keep_par}_${sample_name}_rerun_%a.err" \
            "$baseline_worker_script_step2_cpu" | awk '{print $4}')

        echo "Submitted training baseline array job: ${baseline_cpu_job_id}"
        if [[ -n "$baseline_cpu_job_id" ]]; then
            echo "$baseline_cpu_job_id" >> "$job_id_file"
        fi
    fi
done

