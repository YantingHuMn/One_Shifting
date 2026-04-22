source("../One_Shifting/R/select_best_norm.R")
source("../One_Shifting/R/prep_summary_bubble_table_data.R")
# source("../One_Shifting/R/plot_summary_bubble_table.R")


args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[5]
best_norm_path <- select_best_norm(csv_path)

method <- args[1]
data_type <- args[2]
base_dir <- args[3]
with_col <- toupper(args[4]) %in% c("TRUE", "T", "YES", "1")
norm_info_path <- best_norm_path
out_csv <- args[6]
metric <- args[7]
v2_trans_factor <- args[8]
v2_norm_factor <- args[9]
trans_factors <- args[10:length(args)]


prep_summary_bubble_table_data(method, data_type, base_dir, with_col, norm_info_path, out_csv, metric, v2_trans_factor, v2_norm_factor, trans_factors)

# plot_summary_bubble_table(out_csv)