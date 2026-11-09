# Reverse one peak-level reconstruction, summarize it to gene activity, and score it
# against raw RNA.
#
# Order matters and is fixed here: Step3 writes the reconstruction in TRANSFORMED +
# NORMALIZED peak space, so this script reverses the transformation, then reverses the
# per-cell normalization, and only then sums peaks into gene activity. Summing transformed
# peaks and reversing the gene activity afterwards is not the same operation and is not
# what this does.
#
# RNA enters only at the end. The ATAC side is reconstructed, reversed and summarized on
# its own; the common cell and gene set with RNA is formed once, cached, and reused by
# every combination, so the evaluation set never moves with the data.
#
# The baseline is the SAME q95 peaks with no normalization and no transformation, summed
# through the SAME peak -> gene map. Because nothing about it depends on the combination,
# r_input is identical across all 30 of them; the run prints it so that can be checked.
#
# Parameter:
#   recon_path    : reconstruct_trans_by_<trans>_norm_by_<norm>.feather (cells x peaks)
#   raw_peak_path : Count_Matrix_norm_by_no_norm.feather (cells x peaks, raw q95 counts)
#   lib_sizes_csv : lib_sizes.csv written beside it by prep_peak_q95_matrix.R
#   rna_path      : <sample>_RNA/rna_counts.feather (gene x cell, raw counts)
#   trans, norm   : the combination being scored
#   out_dir       : per-combination output directory
#   cache_dir     : shared across combinations; holds the peak -> gene map, the baseline
#                   activity and the frozen evaluation cell/gene lists
#   method        : method label recorded in the summary row
#   summary_csv   : summary rows are appended here, de-duplicated on
#                   (method, REP1_trans, REP1_norm, corr_type)
#
# Passing PREPARE_ONLY as <recon_path> builds the shared cache (peak -> gene map, baseline
# activity, frozen cell/gene lists, RNA on that set) and exits without scoring anything.
# The runner does that once inside the sample lock, so the three methods cannot race to
# build the same cache; every later call then only reads it.
#   upstream      : gene body plus this many bp upstream, default 2000
# Output: in out_dir, three aligned feathers (ground truth RNA, baseline activity,
#   reconstructed activity), per_gene_metrics_<trans>_<norm>.csv, and one appended row per
#   correlation type in summary_csv.

# The annotation stack is loaded inside build_peak_gene_map() rather than here: the map is
# built once per sample and cached, so 29 of the 30 combinations never need it.
suppressPackageStartupMessages({
    library(arrow)
    library(Matrix)
    library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 10) {
    stop("Usage: Rscript peak_recon_to_gene_activity_eval.R <recon_path> <raw_peak_path> ",
         "<lib_sizes_csv> <rna_path> <trans> <norm> <out_dir> <cache_dir> <method> ",
         "<summary_csv> [upstream]")
}
recon_path    <- args[1]
prepare_only  <- identical(recon_path, "PREPARE_ONLY")
raw_peak_path <- args[2]
lib_sizes_csv <- args[3]
rna_path      <- args[4]
trans         <- args[5]
norm          <- args[6]
out_dir       <- args[7]
cache_dir     <- args[8]
method        <- args[9]
summary_csv   <- args[10]
upstream      <- if (length(args) >= 11 && nzchar(args[11])) as.numeric(args[11]) else 2000

dir.create(out_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------ transformations
# Same definitions as run_correlation_scatter_reverse.R, so the two flows agree.
reverse_trans <- function(mat, trans) {
    if (trans == "no_trans")            return(mat)
    if (trans == "sqrt")                return(mat^2)
    if (trans == "sqrt+1")              return(mat^2 - 1)
    if (trans == "log2")                return(2^mat - 1)
    if (trans == "count+1")             return(mat - 1)
    if (trans == "log2(count+2)")       return(2^mat - 2)
    stop("Unknown trans: ", trans)
}

# forward:  x_norm[i, j] = count[i, j] / lib_size[i] * S   (per cell, rows are cells here)
# reverse:  count[i, j] = x_norm[i, j] / S * lib_size[i]
reverse_norm <- function(mat, norm_factor, lib_sizes) {
    if (norm_factor == "no_norm") return(mat)
    if (norm_factor == "standardize") {
        stop("Cannot reverse 'standardize': the per-peak centers and scales are not saved.")
    }
    S <- suppressWarnings(as.numeric(norm_factor))
    if (is.na(S)) stop("norm factor must be no_norm, standardize or numeric. Got: ", norm_factor)
    if (length(lib_sizes) != nrow(mat)) {
        stop("lib_sizes length (", length(lib_sizes), ") does not match rows (", nrow(mat), ")")
    }
    mat / S * lib_sizes
}

# ------------------------------------------------------------------ io helpers
read_cells_by_features <- function(path, label) {
    if (!file.exists(path)) stop(label, " not found: ", path)
    d <- as.data.frame(read_feather(path))
    if (ncol(d) < 2) stop(label, " must have a cell-id column and feature columns")
    ids <- as.character(d[[1]])
    m <- as.matrix(d[, -1, drop = FALSE])
    storage.mode(m) <- "numeric"
    rownames(m) <- ids
    m
}

assert_finite <- function(m, label) {
    if (anyNA(m))          stop(label, " contains NA")
    if (any(!is.finite(m))) stop(label, " contains Inf or NaN")
    invisible(TRUE)
}

# ------------------------------------------------------------------ peak -> gene map
map_path <- file.path(cache_dir, "peak_gene_map.csv")

build_peak_gene_map <- function(peak_names) {
    suppressPackageStartupMessages({
        library(GenomicRanges)
        library(GenomeInfoDb)
        library(Signac)
        library(EnsDb.Hsapiens.v86)
    })
    peak_df <- do.call(rbind, strsplit(peak_names, "[:-]"))
    if (ncol(peak_df) < 3) stop("peak names must look like chr1-100-200 or chr1:100-200")
    peak_gr <- GRanges(seqnames = peak_df[, 1],
                       ranges = IRanges(start = as.numeric(peak_df[, 2]),
                                        end   = as.numeric(peak_df[, 3])))
    names(peak_gr) <- peak_names

    annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
    old_seq <- seqlevels(annotations)
    new_seq <- ifelse(old_seq == "MT", "chrM", paste0("chr", old_seq))
    names(new_seq) <- old_seq
    annotations <- renameSeqlevels(annotations, new_seq)
    genome(annotations) <- "hg38"

    gene_df <- data.frame(chr = as.character(seqnames(annotations)),
                          start = start(annotations), end = end(annotations),
                          strand = as.character(strand(annotations)),
                          gene_name = annotations$gene_name)
    gene_summary <- gene_df %>%
        dplyr::filter(!is.na(gene_name), gene_name != "") %>%
        dplyr::group_by(gene_name) %>%
        dplyr::summarise(chr = names(sort(table(chr), decreasing = TRUE))[1],
                  start = min(start), end = max(end),
                  strand = names(sort(table(strand), decreasing = TRUE))[1],
                  .groups = "drop")

    gene_coords <- GRanges(
        seqnames = gene_summary$chr,
        ranges = IRanges(
            start = ifelse(gene_summary$strand == "+", gene_summary$start - upstream,
                           gene_summary$start),
            end   = ifelse(gene_summary$strand == "-", gene_summary$end + upstream,
                           gene_summary$end)),
        strand = gene_summary$strand,
        gene_name = gene_summary$gene_name)
    start(gene_coords) <- pmax(start(gene_coords), 1)
    gene_coords <- keepStandardChromosomes(gene_coords, pruning.mode = "coarse")

    ov <- findOverlaps(peak_gr, gene_coords)
    data.frame(peak = peak_names[queryHits(ov)],
               gene = gsub("_", "-", gene_coords$gene_name[subjectHits(ov)]),
               stringsAsFactors = FALSE)
}

# ------------------------------------------------------------------ summarization
# No per-matrix "keep genes with nonzero activity" step: the gene set is the one frozen in
# the cache, so a reconstruction that happens to be zero somewhere cannot shrink it.
summarize_to_activity <- function(peak_mat, pg_map, genes) {
    idx_p <- match(pg_map$peak, colnames(peak_mat))
    keep <- !is.na(idx_p) & pg_map$gene %in% genes
    m <- pg_map[keep, , drop = FALSE]
    idx_p <- idx_p[keep]
    act <- matrix(0, nrow = nrow(peak_mat), ncol = length(genes),
                  dimnames = list(rownames(peak_mat), genes))
    gi <- match(m$gene, genes)
    for (g in unique(gi)) {
        cols <- idx_p[gi == g]
        act[, g] <- if (length(cols) == 1) peak_mat[, cols] else rowSums(peak_mat[, cols, drop = FALSE])
    }
    act
}

write_with_pos <- function(m, path) {
    d <- as.data.frame(m)
    d <- cbind(pos = rownames(m), d)
    rownames(d) <- NULL
    write_feather(d, path)
}

# ------------------------------------------------------------------ fixed map + baseline
raw_mat <- read_cells_by_features(raw_peak_path, "raw q95 peak matrix")
assert_finite(raw_mat, "raw q95 peak matrix")
peak_names <- colnames(raw_mat)

if (!file.exists(map_path)) {
    cat("[EVAL] building the peak -> gene map (once per sample)\n")
    pg_map <- build_peak_gene_map(peak_names)
    write.csv(pg_map, map_path, row.names = FALSE)
} else {
    pg_map <- read.csv(map_path, stringsAsFactors = FALSE)
}
cat(sprintf("[EVAL] peak -> gene map: %d peak-gene pairs over %d peaks\n",
            nrow(pg_map), length(unique(pg_map$peak))))

genes_path <- file.path(cache_dir, "eval_genes.txt")
cells_path <- file.path(cache_dir, "eval_cells.txt")
base_path  <- file.path(cache_dir, "baseline_activity.feather")

rna_eval_path <- file.path(cache_dir, "rna_eval.feather")

# All five pieces, not just the manifest-like ones: a run that died midway can leave some
# of them behind, and treating that as "cached" would silently score against a partial
# evaluation set.
cache_files <- c(map_path, genes_path, cells_path, base_path, rna_eval_path)
cache_complete <- all(file.exists(cache_files))
if (!cache_complete) {
    missing <- basename(cache_files[!file.exists(cache_files)])
    cat(sprintf("[EVAL] evaluation cache incomplete, missing: %s\n",
                paste(missing, collapse = ", ")))
}

if (!cache_complete) {
    cat("[EVAL] freezing the evaluation set from the raw q95 peaks and raw RNA (once)\n")
    rna <- as.data.frame(read_feather(rna_path))
    if (ncol(rna) < 2) stop("RNA file must have a gene-id column and cell columns")
    rna_genes <- gsub("_", "-", as.character(rna[[1]]))
    rna_m <- as.matrix(rna[, -1, drop = FALSE])
    storage.mode(rna_m) <- "numeric"
    rownames(rna_m) <- rna_genes
    assert_finite(rna_m, "raw RNA counts")
    rna_m <- t(rna_m)                                   # cells x genes

    cand_genes <- sort(unique(pg_map$gene))
    base_all <- summarize_to_activity(raw_mat, pg_map, cand_genes)
    # A gene is evaluable when the BASELINE gives it nonzero activity. This depends only on
    # the raw q95 peaks, never on a reconstruction.
    keep_g <- colSums(base_all) > 0
    eval_genes <- base::intersect(cand_genes[keep_g], colnames(rna_m))
    eval_cells <- base::intersect(rownames(raw_mat), rownames(rna_m))
    if (length(eval_genes) == 0) stop("no gene is both nonzero in the baseline activity and present in RNA")
    if (length(eval_cells) == 0) stop("ATAC and RNA share no cell ids")
    eval_genes <- eval_genes[order(match(eval_genes, cand_genes))]
    eval_cells <- eval_cells[order(match(eval_cells, rownames(raw_mat)))]
    writeLines(eval_genes, genes_path)
    writeLines(eval_cells, cells_path)
    write_with_pos(base_all[eval_cells, eval_genes, drop = FALSE], base_path)
    write_with_pos(rna_m[eval_cells, eval_genes, drop = FALSE], rna_eval_path)
    cat(sprintf("[EVAL] frozen: %d cells x %d genes\n", length(eval_cells), length(eval_genes)))
    still_missing <- cache_files[!file.exists(cache_files)]
    if (length(still_missing) > 0) {
        stop("cache build finished but these files are absent: ",
             paste(basename(still_missing), collapse = ", "))
    }
}

if (prepare_only) {
    cat(sprintf("[EVAL] PREPARE_ONLY: cache ready in %s\n", cache_dir))
    quit(save = "no", status = 0)
}

eval_genes <- readLines(genes_path)
eval_cells <- readLines(cells_path)
base_act <- read_cells_by_features(base_path, "baseline activity")
rna_act  <- read_cells_by_features(rna_eval_path, "RNA (evaluation set)")
stopifnot(identical(rownames(base_act), eval_cells), identical(colnames(base_act), eval_genes))
stopifnot(identical(rownames(rna_act),  eval_cells), identical(colnames(rna_act),  eval_genes))

# ------------------------------------------------------------------ this combination
cat(sprintf("[EVAL] %s | trans=%s norm=%s\n", method, trans, norm))
rec <- read_cells_by_features(recon_path, "reconstruction")
assert_finite(rec, "reconstruction")
if (!identical(colnames(rec), peak_names)) {
    stop("the reconstruction's peak columns differ from the fixed q95 peak set ",
         "(", ncol(rec), " vs ", length(peak_names), "). Every combination must use the ",
         "same peaks; re-run prep_peak_q95_matrix.R and Step2/Step3 for this sample.")
}

rec <- reverse_trans(rec, trans)
assert_finite(rec, paste0("reconstruction after reversing '", trans, "'"))

if (norm != "no_norm") {
    lib <- read.csv(lib_sizes_csv, stringsAsFactors = FALSE)
    ls_vec <- lib$lib_size[match(rownames(rec), lib$cell)]
    if (anyNA(ls_vec)) stop(sum(is.na(ls_vec)), " cells have no entry in ", lib_sizes_csv)
    if (any(ls_vec <= 0)) stop("lib_sizes.csv contains non-positive library sizes")
    rec <- reverse_norm(rec, norm, ls_vec)
    assert_finite(rec, paste0("reconstruction after reversing norm '", norm, "'"))
}
cat(sprintf("[EVAL] reversed peaks: min=%.4g max=%.4g (finite negatives are kept as they are)\n",
            min(rec), max(rec)))

rec_act <- summarize_to_activity(rec, pg_map, eval_genes)
rec_act <- rec_act[eval_cells, , drop = FALSE]
assert_finite(rec_act, "reconstructed gene activity")

write_feather(cbind(pos = eval_cells, as.data.frame(rna_act)),
              file.path(out_dir, "eval_ground_truth_rna.feather"))
write_feather(cbind(pos = eval_cells, as.data.frame(base_act)),
              file.path(out_dir, "eval_input_activity.feather"))
write_feather(cbind(pos = eval_cells, as.data.frame(rec_act)),
              file.path(out_dir, "eval_recon_activity.feather"))

# ------------------------------------------------------------------ metrics
cor_by_gene <- function(A, B, meth) {
    vapply(seq_len(ncol(A)), function(j) {
        a <- A[, j]; b <- B[, j]
        if (sd(a) == 0 || sd(b) == 0) return(NA_real_)
        suppressWarnings(cor(a, b, method = meth))
    }, numeric(1))
}

rows <- list()
for (meth in c("pearson", "spearman")) {
    r_input <- cor_by_gene(base_act, rna_act, meth)
    r_recon <- cor_by_gene(rec_act,  rna_act, meth)
    per_gene <- data.frame(
        gene = eval_genes, corr_method = meth,
        r_input = r_input, r_recon = r_recon,
        delta_r = r_input - r_recon,
        improvement = r_recon - r_input,
        input_constant = apply(base_act, 2, sd) == 0,
        recon_constant = apply(rec_act,  2, sd) == 0,
        rna_constant   = apply(rna_act,  2, sd) == 0,
        stringsAsFactors = FALSE)
    rows[[meth]] <- per_gene
    ok <- !is.na(r_input) & !is.na(r_recon)
    cat(sprintf("[EVAL] %-8s genes=%d scored=%d undefined=%d (input_const=%d recon_const=%d rna_const=%d) r_input=%.6f r_recon=%.6f delta_r=%.6f\n",
                meth, length(eval_genes), sum(ok), sum(!ok),
                sum(per_gene$input_constant), sum(per_gene$recon_constant),
                sum(per_gene$rna_constant),
                mean(r_input[ok]), mean(r_recon[ok]), mean((r_input - r_recon)[ok])))
}
per_gene_all <- do.call(rbind, rows)
write.csv(per_gene_all,
          file.path(out_dir, sprintf("per_gene_metrics_%s_norm_%s.csv", trans, norm)),
          row.names = FALSE)

# Column names follow plots_summary_*.csv so run_summary_scatter_plot.R reads this table
# unchanged: x_mean is the mean reconstruction correlation and residual_mean is delta_r.
summary_rows <- do.call(rbind, lapply(c("pearson", "spearman"), function(meth) {
    d <- rows[[meth]]
    ok <- !is.na(d$r_input) & !is.na(d$r_recon)
    data.frame(method = method, REP1_trans = trans, REP1_norm = norm,
               corr_type = meth,
               x_mean = round(mean(d$r_recon[ok]), 6),
               residual_mean = round(mean(d$delta_r[ok]), 6),
               r_input_mean = round(mean(d$r_input[ok]), 6),
               improvement_mean = round(mean(d$improvement[ok]), 6),
               n_genes = length(eval_genes), n_scored = sum(ok), n_undefined = sum(!ok),
               n_input_constant = sum(d$input_constant),
               n_recon_constant = sum(d$recon_constant),
               n_rna_constant = sum(d$rna_constant),
               n_cells = length(eval_cells),
               stringsAsFactors = FALSE)
}))

key <- c("method", "REP1_trans", "REP1_norm", "corr_type")
if (file.exists(summary_csv)) {
    old <- read.csv(summary_csv, stringsAsFactors = FALSE)
    if (!setequal(names(old), names(summary_rows))) {
        stop(summary_csv, " was written with a different column set; delete it and re-run.")
    }
    out <- rbind(old[, names(summary_rows), drop = FALSE], summary_rows)
    dup <- do.call(paste, c(out[, key], sep = "\r"))
    out <- out[!duplicated(dup, fromLast = TRUE), , drop = FALSE]
} else {
    dir.create(dirname(summary_csv), recursive = TRUE, showWarnings = FALSE)
    out <- summary_rows
}
write.csv(out, summary_csv, row.names = FALSE)
cat(sprintf("[EVAL] summary -> %s (%d rows)\n", summary_csv, nrow(out)))