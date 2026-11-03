# Peak-level MSE self-reconstruction: one sample, one method.
# folder_name / sample_name / input_path may be exported by an upstream script; the
# values below are only the fallback for running this by hand.

folder_name="${folder_name:-HCA}"
sample_name="${sample_name:-Hrv39}"

OUT_DIR="/dcs11/hongkai/data/yhu1/One_Shifting_Results"
READ_DIR="${OUT_DIR}/${folder_name}/${sample_name}_INPUT_ATAC_peak_MSE"

V1="${sample_name}_ATAC"
V2="${sample_name}_RNA"

# Raw h5ad for this sample. Looked up from the folder's TSV by sample_name when the
# upstream script did not already provide it; the lookup must match exactly one row.
if [ -z "${input_path}" ]; then
    TSV="../One_Shifting/data/${folder_name}_input_sample_pairs.tsv"
    if [ ! -f "$TSV" ]; then
        echo "[CONFIG] sample table not found: $TSV" >&2
        return 1 2>/dev/null || exit 1
    fi
    n_hit=$(awk -F'\t' -v s="$sample_name" 'NR>1 && $2==s' "$TSV" | wc -l)
    if [ "$n_hit" -ne 1 ]; then
        echo "[CONFIG] sample_name '${sample_name}' matched ${n_hit} rows in $TSV (need exactly 1)" >&2
        return 1 2>/dev/null || exit 1
    fi
    input_path=$(awk -F'\t' -v s="$sample_name" 'NR>1 && $2==s {print $1; exit}' "$TSV")
fi

# Peak counts the models see, and the raw RNA used only at evaluation.
INPUT_FILE="${OUT_DIR}/${folder_name}/${V1}/atac_peak_counts_top_q95.feather"
GROUND_TRUTH_FILE="${OUT_DIR}/${folder_name}/${V2}/rna_counts.feather"

V2_norm_factor="no_norm"
V2_trans_factor="no_trans"

GENOME_BUILD="hg38"