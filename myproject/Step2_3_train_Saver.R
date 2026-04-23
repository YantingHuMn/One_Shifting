load_or_install <- function(pkg, bioc = FALSE) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        if (bioc) {
            if (!requireNamespace("BiocManager", quietly = TRUE)) {
                install.packages("BiocManager")
            }
            BiocManager::install(pkg)
        } else {
            install.packages(pkg)
        }
    }
    library(pkg, character.only = TRUE)
}

load_or_install("arrow")
load_or_install("SAVER", bioc = TRUE)
load_or_install("Matrix")

args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_path <- args[2]
ncores <- if (length(args) >= 3) as.integer(args[3]) else 1

df <- read_feather(input_path)
cat("Input shape:", dim(df), "\n")

pos_col <- df[[1]]
mat <- as.matrix(df[, -1])
cat("Matrix shape:", dim(mat), "\n")

# SAVER require gene x cell
mat_t <- t(mat)  # transpose to gene x cell
mat_t <- as(mat_t, "dgCMatrix")  # to sparse df
cat("Transposed shape (gene x cell):", dim(mat_t), "\n")

# Run SAVER
cat("Running SAVER...\n")
saver_result <- saver(mat_t, ncores = ncores)
denoised_mat <- saver_result$estimate  # gene x cell
cat("SAVER output shape (gene x cell):", dim(denoised_mat), "\n")

# Transpose back to cell x gene
denoised_mat <- t(denoised_mat)  # cell x gene
cat("Output shape (cell x gene):", dim(denoised_mat), "\n")

# Save and output df
out_df <- data.frame(pos = pos_col, denoised_mat, check.names = FALSE)
colnames(out_df)[-1] <- colnames(df)[-1]

write_feather(out_df, output_path)
cat("Done! Output saved to:", output_path, "\n")
cat("Output shape:", dim(out_df), "\n")