echo "DEBUG sample_name='$sample_name'"
echo "Folder name='$folder_name'"
echo "dropout_keep_par='$dropout_keep_par'"

OUT_DIR="/dcs07/hongkai/data/yhu1/One_Shifting_Results"
RAW_READ_DIR="${OUT_DIR}/${folder_name}/${sample_name}_INPUT_ATAC"

if [[ -n "$dropout_keep_par" ]]; then
    dropout_tag=$(echo "$dropout_keep_par" | sed 's/\./p/g')
    READ_DIR="${RAW_READ_DIR}/dropout_${dropout_tag}"
else
    READ_DIR="${RAW_READ_DIR}"
fi
V1="${sample_name}_ATAC"
V2="${sample_name}_RNA"
INPUT_FILE="${OUT_DIR}/${folder_name}/${V1}/activity_counts.feather"
GROUND_TRUTH_FILE="${OUT_DIR}/${folder_name}/${V2}/rna_counts.feather"
V2_norm_factor="no_norm"
V2_trans_factor="no_trans"