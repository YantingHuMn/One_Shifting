source("/dcs10/hongkai/data/yhu1/One_Shifting/R/summarize_peak_to_gene_activity.R")

args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_recon_path <- args[2]
INPUT_TRANS_PATH <- args[3]
INPUT_TRANS_PATH_filtered <-args[4]
GROUND_TRUTH_path <- args[5]
filtered_GROUND_TRUTH_path <- args[6]

summarize_peak_to_gene_activity(input_path = input_path, output_recon_path = output_recon_path, INPUT_TRANS_PATH = INPUT_TRANS_PATH, INPUT_TRANS_PATH_filtered = INPUT_TRANS_PATH_filtered, GROUND_TRUTH_path = GROUND_TRUTH_path, filtered_GROUND_TRUTH_path = filtered_GROUND_TRUTH_path)
