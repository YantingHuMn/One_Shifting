#!/bin/bash
#SBATCH --job-name=multi_One_Shifting
#SBATCH --partition=cpu
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --output=/projects/foundation_model_for_single_cell_multiomics_data/yhu157/logs/submit_multi.out
#SBATCH --error=/projects/foundation_model_for_single_cell_multiomics_data/yhu157/logs/submit_multi.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
# #SBATCH --dependency=afterok:31582287

my_dir="/home/yhu157/Autoencoder"
project_dir="/projects/foundation_model_for_single_cell_multiomics_data/data_processed/scMultiomics"
folder_name="HCA"
out_dir="/projects/foundation_model_for_single_cell_multiomics_data/yhu157"

mkdir -p "${out_dir}/logs"

# Step 1: Find .h5ad only in immediate subfolders of HCA
h5ad_list="${out_dir}/${folder_name}_h5ad_files.txt"

> "$h5ad_list"

for subdir in "${project_dir}/${folder_name}/"*; do
    if [[ -d "$subdir" ]]; then

        tmp_file=$(mktemp)

        find "$subdir" \
            -maxdepth 1 \
            -type f \
            -name "*_DHS_blacklist_rm_noY.h5ad" \
            | sort > "$tmp_file"

        n=$(wc -l < "$tmp_file")

        # If this subfolder has 3 or more files, only keep the first 2.
        # If it has 1 or 2 files, keep all.
        if [[ "$n" -ge 3 ]]; then
            head -n 2 "$tmp_file" >> "$h5ad_list"
        else
            cat "$tmp_file" >> "$h5ad_list"
        fi

        rm "$tmp_file"
    fi
done

echo "Number of selected h5ad files:"
wc -l "$h5ad_list"

# Step 2: Generate Sample Name
pair_tsv="${out_dir}/${folder_name}_input_sample_pairs.tsv"

echo -e "input_path\tsample_name" > "$pair_tsv"

while read -r input_path; do
    fname=$(basename "$input_path")

    sample_name="$fname"
    sample_name="${sample_name#multi_}"
    sample_name="${sample_name#atac_}"
    sample_name="${sample_name%_DHS_blacklist_rm_noY.h5ad}"
    sample_name="${sample_name%_DHS_blacklist_rm_Y.h5ad}"

    sample_name="${sample_name#PMID_}"
    sample_name=$(echo "$sample_name" | sed -E 's/^[0-9]+_//')

    echo -e "${input_path}\t${sample_name}"
done < "$h5ad_list" >> "$pair_tsv"

echo "First few input/sample pairs:"
head "$pair_tsv"

# Step 3: Create Count Matrix
module load anaconda3/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate r_anndata

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

module load R/4.4.0+Bioconductor

export RETICULATE_PYTHON="$(which python)"

echo "Using Python:"
which python
python -c "import anndata, h5py, numpy; print('anndata', anndata.__version__); print('h5py', h5py.__version__); print('numpy', numpy.__version__)"

echo "Using Rscript:"
which Rscript
Rscript --version

tail -n +2 "$pair_tsv" | while IFS=$'\t' read -r input_path sample_name; do
    if [[ -z "$input_path" || -z "$sample_name" ]]; then
        echo "Skipping empty line"
        continue
    fi

    echo "======================================"
    echo "Running:"
    echo "input_path=${input_path}"
    echo "sample_name=${sample_name}"
    echo "out_dir=${out_dir}"
    echo "======================================"

    Rscript "${my_dir}/One_Shifting/R/load_rna_atac_from_h5ad.R" \
        "$input_path" \
        "$sample_name" \
        "$out_dir"

    status=$?

    if [[ "$status" -ne 0 ]]; then
        echo "ERROR: Rscript failed for sample_name=${sample_name}"
        echo "Exit status: $status"
        exit "$status"
    fi
done