source("../One_Shifting/R/plot_summary_bubble_table.R")

args <- commandArgs(trailingOnly = TRUE)

csv_path <- args[1]

if (length(args) >= 2) {
    filter_method <- args[2]
    plot_summary_bubble_table(csv_path, filter_method)
} else {
    plot_summary_bubble_table(csv_path)
}
