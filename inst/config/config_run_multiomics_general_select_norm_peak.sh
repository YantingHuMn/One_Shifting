
folder_name="HCA"
sample_name="G120_D_TL"

OUT_DIR="/dcs07/hongkai/data/yhu1/One_Shifting_Results"
READ_DIR="${OUT_DIR}/${folder_name}/${sample_name}_INPUT_ATAC_peak_BCE"
V1="${sample_name}_ATAC"
V2="${sample_name}_RNA"
INPUT_FILE="${OUT_DIR}/${folder_name}/${V1}/atac_peak_counts_top_q95.feather"
GROUND_TRUTH_FILE="${OUT_DIR}/${folder_name}/${V2}/rna_counts.feather"
V2_norm_factor="no_norm"
V2_trans_factor="no_trans"

