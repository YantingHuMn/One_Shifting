# Produce the q95 peak counts WITHOUT touching anything else.
#
# load_rna_atac_from_h5ad.R writes rna_counts.feather, activity_counts.feather and
# atac_peak_counts.feather in the same run as the q95 file. The gene-activity experiments
# already depend on those, so the peak flow must never call it just to obtain q95. This
# script writes exactly one file: the q95 peak counts at --out_path.
#
# Source, in order of preference:
#   1. an existing atac_peak_counts.feather (peak x cell) -- no h5ad is read
#   2. the h5ad, from which ONLY the Chromatin Accessibility features are taken
#
# The selection rule is the one already in load_rna_atac_from_h5ad.R: a ChromatinAssay
# over the non-empty peaks, RunTFIDF, then FindTopFeatures(min.cutoff = "q95"), i.e. the
# peaks whose total count across cells is above the 95th percentile. The matrix stays
# sparse throughout; only the selected ~5% is made dense, to be written as a feather.
#
# Parameter:
#   --out_path    where to write atac_peak_counts_top_q95.feather (required)
#   --peak_counts existing atac_peak_counts.feather, used when present
#   --h5ad        raw h5ad, used only when --peak_counts is absent
# Output: --out_path only. Never writes RNA, activity, or any file under an existing
#   gene-activity experiment directory.

suppressPackageStartupMessages({
    library(arrow)
    library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(k, required = FALSE) {
    i <- which(args == paste0("--", k))
    if (length(i) == 1 && i < length(args)) return(args[i + 1])
    if (required) stop("missing required argument --", k)
    NULL
}
out_path    <- get_arg("out_path", required = TRUE)
peak_counts <- get_arg("peak_counts")
h5ad_path   <- get_arg("h5ad")

if (file.exists(out_path)) {
    cat(sprintf("[Q95] %s already exists; leaving it untouched\n", out_path))
    quit(save = "no", status = 0)
}
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

select_q95 <- function(atac_mat) {
    suppressPackageStartupMessages({
        library(Signac)
        library(GenomicRanges)
        library(Seurat)
    })
    keep <- Matrix::rowSums(atac_mat > 0) > 0
    cat(sprintf("[Q95] peaks non-zero in at least one cell: %d / %d\n", sum(keep), length(keep)))
    atac_mat <- atac_mat[keep, , drop = FALSE]

    peak_names <- rownames(atac_mat)
    pdf_ <- do.call(rbind, strsplit(peak_names, "[:-]"))
    if (ncol(pdf_) < 3) stop("peak names must look like chr1-100-200 or chr1:100-200")
    peak_gr <- GRanges(seqnames = pdf_[, 1],
                       ranges = IRanges(start = as.numeric(pdf_[, 2]),
                                        end   = as.numeric(pdf_[, 3])))
    names(peak_gr) <- peak_names

    assay <- CreateChromatinAssay(counts = atac_mat, ranges = peak_gr)
    obj <- Seurat::CreateSeuratObject(counts = assay, assay = "peaks")
    obj <- RunTFIDF(obj, assay = "peaks")
    obj <- FindTopFeatures(obj, assay = "peaks", min.cutoff = "q95")
    top <- Seurat::VariableFeatures(obj)
    if (length(top) == 0) stop("FindTopFeatures(min.cutoff='q95') selected no peaks")
    cat(sprintf("[Q95] selected %d peaks (%.1f%% of %d)\n",
                length(top), 100 * length(top) / nrow(atac_mat), nrow(atac_mat)))
    # only the selected peaks are made dense
    atac_mat[top, , drop = FALSE]
}

if (!is.null(peak_counts) && file.exists(peak_counts)) {
    cat(sprintf("[Q95] source: existing peak counts %s\n", peak_counts))
    d <- read_feather(peak_counts)
    ids <- as.character(d[[1]])
    cell_names <- names(d)[-1]
    # One cell column at a time: only its non-zero entries are kept, so the full
    # peak x cell block is never held as a dense matrix.
    i_all <- integer(0); j_all <- integer(0); x_all <- numeric(0)
    for (k in seq_along(cell_names)) {
        v <- as.numeric(d[[k + 1]])
        nz <- which(v != 0)
        if (length(nz) > 0) {
            i_all <- c(i_all, nz)
            j_all <- c(j_all, rep.int(k, length(nz)))
            x_all <- c(x_all, v[nz])
        }
    }
    atac_mat <- sparseMatrix(i = i_all, j = j_all, x = x_all,
                             dims = c(length(ids), length(cell_names)),
                             dimnames = list(ids, cell_names))
    rm(d, i_all, j_all, x_all)
} else {
    if (is.null(h5ad_path) || !file.exists(h5ad_path)) {
        stop("neither --peak_counts nor a readable --h5ad was given; cannot build q95.\n",
             "  --peak_counts: ", if (is.null(peak_counts)) "(not set)" else peak_counts, "\n",
             "  --h5ad       : ", if (is.null(h5ad_path)) "(not set)" else h5ad_path)
    }
    cat(sprintf("[Q95] source: h5ad %s (ATAC features only)\n", h5ad_path))
    suppressPackageStartupMessages(library(anndata))
    adata <- read_h5ad(h5ad_path)
    ft <- adata$var$feature_types
    atac_idx <- which(ft == "Chromatin Accessibility")
    if (length(atac_idx) == 0) {
        stop("no var rows with feature_types == 'Chromatin Accessibility' in ", h5ad_path)
    }
    # subset to ATAC before transposing, so the RNA block is never materialised here
    sub <- adata$X[, atac_idx, drop = FALSE]
    atac_mat <- as(t(sub), "sparseMatrix")
    rownames(atac_mat) <- adata$var_names[atac_idx]
    colnames(atac_mat) <- adata$obs_names
    rm(sub, adata)
}

cat(sprintf("[Q95] input peak matrix: %d peaks x %d cells\n", nrow(atac_mat), ncol(atac_mat)))
if (any(atac_mat@x < 0)) stop("peak counts contain negative values; this must be raw counts")

top_mat <- as.matrix(select_q95(atac_mat))
out <- cbind(data.frame(pos = rownames(top_mat), stringsAsFactors = FALSE),
             as.data.frame(top_mat))
rownames(out) <- NULL

# write via a temporary name so a reader can never see a half-written file
tmp_path <- paste0(out_path, ".tmp")
write_feather(out, tmp_path)
if (!file.rename(tmp_path, out_path)) stop("could not move ", tmp_path, " to ", out_path)
cat(sprintf("[Q95] wrote %s : %d peaks x %d cells\n", out_path, nrow(top_mat), ncol(top_mat)))