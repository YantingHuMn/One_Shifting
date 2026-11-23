#!/bin/bash
cd /dcs10/hongkai/data/yhu1/One_Shifting
read_dir="/dcs11/hongkai/data/yhu1/One_Shifting_Results"

export folder_name="10x"

# choose_combination: Transformer_denoise transformation x normalization selection (ShiftFormer package)
tasks="choose_combination"
has_task () { [[ ",${tasks// /}," == *",$1,"* ]]; }

# Kept fraction of each cell's library; 1 = no dropout (raw matrix used as the dropout input).
dropout_keep_par_values=(0.7 0.8 0.9 1) # keep

# DRY_RUN=1 bash submit_package.sh -> print the sbatch commands without submitting
DRY_RUN="${DRY_RUN:-0}"

shiftformer_worker_script="../One_Shifting/sbatch/task/run_ShiftFormer_choose_combination_array.sh"
tsv="../One_Shifting/data/${folder_name}_input_sample_pairs.tsv"

log_dir="${read_dir}/${folder_name}/AAA_logs"
mkdir -p "$log_dir"

# Samples (row numbers in the tsv, header excluded)
if [[ "$folder_name" == "HCA" ]]; then
    selected_rows="4"
elif [[ "$folder_name" == "ENCODE" ]]; then
    selected_rows="1"
elif [[ "$folder_name" == "10x" ]]; then
    selected_rows="1"
fi

for dropout_keep_par in "${dropout_keep_par_values[@]}"; do

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

        # ShiftFormer: array 0 = INPUT_RNA, 1 = INPUT_ATAC
        if has_task choose_combination; then
            echo "Submitting ShiftFormer array for sample_name=${sample_name}, dropout_keep_par=${dropout_keep_par}"

            sbatch_cmd=(sbatch
                --job-name="SF_${folder_name}_${dropout_keep_par}_${sample_name}"
                --exclude=compute-170,compute-128
                --partition=gpu
                --gres=gpu:1
                --time=3-00:00:00
                --mem=128G
                --cpus-per-task=3
                --array=0-1
                --export=ALL,folder_name="${folder_name}",sample_name="${sample_name}",dropout_keep_par="${dropout_keep_par}"
                --output="${log_dir}/submit_${folder_name}_ShiftFormer_${dropout_keep_par}_${sample_name}_%a.out"
                --error="${log_dir}/submit_${folder_name}_ShiftFormer_${dropout_keep_par}_${sample_name}_%a.err"
                "$shiftformer_worker_script")

            if [[ "$DRY_RUN" == "1" ]]; then
                echo "[DRY_RUN] ${sbatch_cmd[*]}"
            else
                sf_job_id=$("${sbatch_cmd[@]}" | awk '{print $4}')
                echo "Submitted ShiftFormer array job: ${sf_job_id}"
            fi
        fi

    done

done
