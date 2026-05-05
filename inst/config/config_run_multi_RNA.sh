echo "DEBUG sample_name='$sample_name'"

READ_DIR="../One_Shifting_Results/${sample_name}_INPUT_RNA"
V1="${sample_name}_RNA"
V2="${sample_name}_ATAC"
INPUT_FILE="../One_Shifting_Results/${V1}/rna_counts.feather"
GROUND_TRUTH_FILE="../One_Shifting_Results/${V2}/activity_counts.feather"
V2_norm_factor="no_norm"
V2_trans_factor="no_trans"