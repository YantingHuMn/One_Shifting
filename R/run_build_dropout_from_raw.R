# Build Dropout Matrix From Raw (ShiftFormer choose_combination input)
# Post: Apply dropout_multinomial() directly to an already-built raw count matrix (cell x gene,
#       first column = cell ID). No gene or cell filtering is redone, so the dropout matrix has exactly
#       the same cells and genes, in the same order, as the raw matrix.
# Parameter:
#   raw_path : raw Count_Matrix_norm_by_no_norm.feather
#   out_path : output feather path for the dropout matrix
#   keep_par : fraction of each cell's library kept, 0 < keep_par < 1 (1 = no dropout, handled by caller)
#   seed     : random seed set before dropout
# Output: out_path, written to a temporary file first and then renamed (never a partial file).
source("../One_Shifting/R/dropout_multinomial.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
    stop("Usage: Rscript run_build_dropout_from_raw.R <raw_path> <out_path> <keep_par> <seed>")
}
raw_path <- args[1]
out_path <- args[2]
keep_par <- as.numeric(args[3])
seed     <- as.integer(args[4])
if (is.na(keep_par) || keep_par <= 0 || keep_par >= 1) {
    stop("keep_par must be in (0, 1). Got: ", args[3])
}

suppressPackageStartupMessages({
    library(arrow)
})

cat("\n=== Reading raw matrix ===\n", raw_path, "\n")
raw <- as.data.frame(read_feather(raw_path), check.names = FALSE)
cell_ids <- as.character(raw[[1]])
gene_ids <- colnames(raw)[-1]
if (anyNA(cell_ids) || anyDuplicated(cell_ids) || anyNA(gene_ids) || anyDuplicated(gene_ids)) {
    stop("Raw matrix cell/gene IDs contain missing or duplicate values.")
}
cat(sprintf("  %d cells x %d genes\n", nrow(raw), length(gene_ids)))

cat(sprintf("\n=== Applying multinomial dropout: keep_par = %s, seed = %d ===\n", keep_par, seed))
set.seed(seed)
drop <- dropout_multinomial(raw, keep_par = keep_par, cell_axis = "row")
colnames(drop)[1] <- colnames(raw)[1]
if (!identical(colnames(drop), colnames(raw)) || !identical(as.character(drop[[1]]), cell_ids) ||
    !identical(dim(drop), dim(raw))) {
    stop("Dropout matrix does not keep the raw cell/gene IDs and order.")
}
tot_raw <- sum(as.matrix(raw[, -1])); tot_drop <- sum(as.matrix(drop[, -1]))
cat(sprintf("  total counts: raw = %.0f, dropout = %.0f (ratio %.4f); zero-library cells = %d\n",
            tot_raw, tot_drop, tot_drop / tot_raw, sum(rowSums(as.matrix(drop[, -1])) == 0)))

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
tmp_path <- paste0(out_path, ".tmp_", Sys.getpid())
write_feather(drop, tmp_path)
if (!file.rename(tmp_path, out_path)) {
    stop("Could not rename ", tmp_path, " to ", out_path)
}
cat("\n=== Saved dropout matrix ===\n", out_path, "\n")
