# Prepare the normalized peak matrices for the peak-level MSE self-reconstruction run.
#
# Post: Takes the q95 peak counts written by load_rna_atac_from_h5ad.R (peak x cell, first
#   column `pos` = peak id), drops the rows and columns that would make the normalization
#   undefined, and writes one Count_Matrix_norm_by_<factor>.feather per normalization plus
#   lib_sizes.csv -- the same layout Step2/Step3 already read for the gene-activity runs.
#   Normalization is done by normalize_and_save.R, which is also what the gene-activity
#   path uses, so the two flows divide by the same per-cell totals.
#
#   Step1_build_count_matrix.R is deliberately NOT used here: its lib_size_min cut would
#   remove peaks on top of the q95 selection, and the whole point of the q95 file is that
#   one fixed peak set is shared by every model and every (transformation, normalization)
#   combination.
#
# Parameter:
#   peak_path  : .../<sample>_ATAC/atac_peak_counts_top_q95.feather (peak x cell)
#   out_dir    : READ_DIR; matrices land in <out_dir>/<sub_dir_name>/
#   sub_dir_name : usually "<sample>_ATAC"
#   norm_string  : comma separated normalizations, e.g. "no_norm,1000000,100000,10000,1000"
#   genome_build : recorded in the manifest, e.g. "hg38"
# Output: <out_dir>/<sub_dir_name>/Count_Matrix_norm_by_<factor>.feather (cells x peaks,
#   first column `pos` = cell id), lib_sizes.csv, and peak_cell_manifest.txt holding the
#   fixed peak and cell names plus the genome build.

suppressPackageStartupMessages({
    library(arrow)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
    stop("Usage: Rscript prep_peak_q95_matrix.R <peak_path> <out_dir> <sub_dir_name> ",
         "<norm_string> [genome_build]")
}
peak_path    <- args[1]
out_dir      <- args[2]
sub_dir_name <- args[3]
norm_string  <- args[4]
genome_build <- if (length(args) >= 5 && nzchar(args[5])) args[5] else "hg38"

# The q95 block in load_rna_atac_from_h5ad.R sits inside a tryCatch that only warns, so a
# missing file here means that step failed rather than that it was never run. Say so.
if (!file.exists(peak_path)) {
    stop("q95 peak file not found: ", peak_path, "\n",
         "  load_rna_atac_from_h5ad.R writes it inside a tryCatch that only prints a ",
         "warning, so check that step's log for 'failed to save atac_peak_counts_top_q95' ",
         "before re-running this.")
}

source("../One_Shifting/R/normalize_and_save.R")

cat(sprintf("[PREP] reading %s\n", peak_path))
df <- as.data.frame(read_feather(peak_path))
if (ncol(df) < 2) stop("q95 peak file must have one peak-id column and cell columns")
peak_ids <- as.character(df[[1]])
mat <- as.matrix(df[, -1, drop = FALSE])
storage.mode(mat) <- "numeric"
rownames(mat) <- peak_ids
cat(sprintf("[PREP] q95 matrix: %d peaks x %d cells\n", nrow(mat), ncol(mat)))

if (anyNA(mat)) stop("q95 peak counts contain NA")
if (any(!is.finite(mat))) stop("q95 peak counts contain non-finite values")
if (any(mat < 0)) stop("q95 peak counts contain negative values; this file must be raw counts")

# A cell whose peaks all sum to zero has no library size to divide by, and a peak that is
# zero in every cell carries no signal. Both are removed HERE, once, so the peak and cell
# sets are fixed for every combination downstream rather than varying with the data.
cell_tot <- colSums(mat)
peak_tot <- rowSums(mat)
drop_cells <- which(cell_tot == 0)
drop_peaks <- which(peak_tot == 0)
if (length(drop_cells) > 0) {
    cat(sprintf("[PREP] dropping %d cell(s) with zero total peak counts (division by zero)\n",
                length(drop_cells)))
}
if (length(drop_peaks) > 0) {
    cat(sprintf("[PREP] dropping %d peak(s) with zero counts in every cell\n",
                length(drop_peaks)))
}
if (length(drop_cells) > 0) mat <- mat[, -drop_cells, drop = FALSE]
if (length(drop_peaks) > 0) mat <- mat[-drop_peaks, , drop = FALSE]
if (ncol(mat) == 0) stop("no cells remain after removing zero-library-size cells")
if (nrow(mat) == 0) stop("no peaks remain after removing all-zero peaks")
cat(sprintf("[PREP] fixed peak set: %d peaks x %d cells\n", nrow(mat), ncol(mat)))

count_data <- cbind(data.frame(pos = rownames(mat), stringsAsFactors = FALSE),
                    as.data.frame(mat))
rownames(count_data) <- NULL

norm_factors <- trimws(strsplit(norm_string, ",")[[1]])
norm_factors <- norm_factors[nzchar(norm_factors)]
cat(sprintf("[PREP] normalizations: %s\n", paste(norm_factors, collapse = ", ")))

# transpose = TRUE so the saved matrices are cells x peaks with `pos` = cell id, which is
# the orientation Step2/Step3 expect.
normalize_and_save(count_data, out_dir, sub_dir_name, norm_factors,
                   keep_rows = NULL, transpose = TRUE)

save_dir <- file.path(out_dir, sub_dir_name)
writeLines(c(sprintf("# genome_build: %s", genome_build),
             sprintf("# source: %s", peak_path),
             sprintf("# n_peaks: %d", nrow(mat)),
             sprintf("# n_cells: %d", ncol(mat)),
             "# peaks:",
             rownames(mat),
             "# cells:",
             colnames(mat)),
           file.path(save_dir, "peak_cell_manifest.txt"))

for (nf in norm_factors) {
    p <- file.path(save_dir, paste0("Count_Matrix_norm_by_", nf, ".feather"))
    if (!file.exists(p)) stop("expected output missing: ", p)
    d <- read_feather(p)
    cat(sprintf("[PREP] wrote %s : %d cells x %d peaks\n", basename(p), nrow(d), ncol(d) - 1))
}
cat(sprintf("[PREP] manifest: %s\n", file.path(save_dir, "peak_cell_manifest.txt")))