source("../One_Shifting/R/plot_summary_bubble_table.R")

args <- commandArgs(trailingOnly = TRUE)

csv_path <- args[1] 

plot_summary_bubble_table(csv_path)
