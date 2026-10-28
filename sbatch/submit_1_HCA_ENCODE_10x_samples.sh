#!/bin/bash
cd /dcs10/hongkai/data/yhu1/One_Shifting
read_dir="/dcs11/hongkai/data/yhu1/One_Shifting_Results"

export folder_name="HCA"

# train, baseline step2_only
tasks="train"
has_task () { [[ ",${tasks// /}," == *",$1,"* ]]; }

# Select 5 representative samples from diverse tissues to assess generalizability.
# 0.3, 0.5, 0.7, 0.9
dropout_keep_par_values="0.3,0.5,0.7,0.9" #keep

worker_script="../One_Shifting/sbatch/task/run_HCA_ENCODE_10x_one_sample_array.sh"
baseline_worker_script_step1_gpu="../One_Shifting/sbatch/task/run_scVI_DCA_Saver_multi_step1.sh"
baseline_worker_script_step2_cpu="../One_Shifting/sbatch/task/run_scVI_DCA_Saver_multi_step2.sh"
tsv="${read_dir}/${folder_name}_input_sample_pairs.tsv"

log_dir="${read_dir}/${folder_name}/AAA_logs"
mkdir -p "$log_dir"
job_id_file="${log_dir}/AAA_step2_job_map.txt"
touch "$job_id_file"

# run_first_n_sample=4
# tail -n +2 "$tsv" | awk -v n="$run_first_n_sample" 'NR <= n {print $2}' | while read -r sample_name; do
# selected_rows="1,2,3"
# selected_rows="4,5,6,7,8,9"
# selected_rows="10,11,12,13"
# selected_rows="14,15,16,17,18,19"
# selected_rows="10,11,12,13,14,15,16,17,18,19"

# dropout Samples
if [[ "$folder_name" == "HCA" ]]; then
    selected_rows="4,15"
elif [[ "$folder_name" == "ENCODE" ]]; then
    selected_rows="1,5"
elif [[ "$folder_name" == "10x" ]]; then
    selected_rows="1"
fi

for dropout_keep_par in ${dropout_keep_par_values//,/ }; do

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

            methods_par="VAE DCA_mse"

            train_job_id=$(sbatch \
                --job-name="${folder_name}_${dropout_keep_par}_${sample_name}" \
                --exclude=compute-170 \
                --partition=gpu \
                --gres=gpu:1 \
                --time=3-00:00:00 \
                --mem=80G \
                --cpus-per-task=1 \
                --array=0-1 \
                --export=ALL,folder_name="${folder_name}",sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}",methods_par="${methods_par}" \
                --output="${log_dir}/submit_${folder_name}_gpu_${dropout_keep_par}_${sample_name}_%a.out" \
                --error="${log_dir}/submit_${folder_name}_gpu_${dropout_keep_par}_${sample_name}_%a.err" \
                "$worker_script" | awk '{print $4}')

            echo "Submitted training array job: ${train_job_id}"
        fi

        if has_task train; then
            echo "Submitting training array for sample_name=${sample_name}"

            methods_par="Transformer_denoise"

            train_job_id=$(sbatch \
                --job-name="${folder_name}_${dropout_keep_par}_${sample_name}" \
                --exclude=compute-170 \
                --partition=shared \
                --time=7-00:00:00 \
                --mem=80G \
                --cpus-per-task=1 \
                --array=0 \
                --export=ALL,folder_name="${folder_name}",sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}",methods_par="${methods_par}" \
                --output="${log_dir}/submit_${folder_name}_cpu_${dropout_keep_par}_${sample_name}_%a.out" \
                --error="${log_dir}/submit_${folder_name}_cpu_${dropout_keep_par}_${sample_name}_%a.err" \
                "$worker_script" | awk '{print $4}')

            echo "Submitted training array job: ${train_job_id}"
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

done