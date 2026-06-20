
folder_name="HCA"
sample_name="G120_D_TL"

OUT_DUR="/dcs07/hongkai/data/yhu1/One_Shifting_Results"
READ_DIR="${OUT_DUR}/${folder_name}/${sample_name}_INPUT_ATAC"
V1="${sample_name}_ATAC"
V2="${sample_name}_RNA"
INPUT_FILE="${OUT_DUR}/${folder_name}/${V1}/atac_peak_counts_top_q90.feather"
V2_norm_factor="no_norm"
V2_trans_factor="no_trans"