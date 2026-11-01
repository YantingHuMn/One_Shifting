# Preparation for the two MLP tasks
#
# Post: Builds, for one experiment, a fixed cell split and fixed gene lists, then extracts
#   aligned input/target matrices for one (transformation, normalization) combination.
#   Cell identifiers are carried in the `pos` column of every file it writes, so the
#   Python side can align by name rather than by position.
#
# Two modes, because the split and the gene lists must be shared by every
# (transformation, normalization) run of the same experiment:
#
#   --mode lists    run once per experiment, on the raw-count (no_norm) matrices.
#                   Order: align by cell id -> fix train/val/test -> select genes using
#                   TRAINING CELLS ONLY -> write the TXT lists. Test cells never enter
#                   gene filtering or HVG selection.
#   --mode extract  run once per (transformation, normalization). Reads the split and the
#                   TXT lists written by `lists` and slices the normalized matrices by
#                   them. Selects nothing.
#
# Two tasks:
#   --task within   input and target are the same modality. Input genes are the TFs found
#                   in that matrix; target genes are non-TF HVGs of the same matrix.
#   --task cross    input is the ATAC gene-activity matrix, target is the RNA matrix of
#                   the same cells, over ONE SHARED gene set: the non-TF RNA HVGs that
#                   also survive the training-cell filter on the ATAC side, kept in HVG
#                   order. X is the ATAC activity of those genes and Y is the RNA
#                   expression of the same genes in the same cells, so the two matrices
#                   have identical column names in identical order. Cells are paired by
#                   `pos`. n_hvg is the number of RNA HVG CANDIDATES; the intersection is
#                   usually smaller and is never padded back up.
#
# Parameter (all --key value):
#   --mode             lists | extract
#   --task             within | cross
#   --input_matrix     cells x genes feather, first column `pos`
#   --target_matrix    cells x genes feather; equal to --input_matrix for `within`
#   --out_dir          where the six matrices for this run are written
#   --lists_dir        where the split and the TXT gene lists live (written by `lists`,
#                      read by `extract`); shared across trans/norm
#   --tf_list          TF annotation csv (default: the repo's DatabaseExtract copy)
#   --n_hvg            HVGs requested from Seurat VST (default 1000)
#   --min_cells        a gene is kept when it is non-zero in at least this many TRAINING
#                      cells (default 3)
#   --test_frac        default 0.2
#   --val_frac         default 0.1
#   --seed             default 42
#   --input_genes      optional TXT overriding the input gene list
#   --target_genes     optional TXT overriding the target gene list
# Output:
#   mode lists  : <lists_dir>/cell_split.csv, tf_names.txt, hvg_target_names.txt,
#                 input_genes.txt, target_genes.txt, prep_meta.json
#   mode extract: <out_dir>/input_{train,val,test}.feather,
#                 <out_dir>/target_{train,val,test}.feather   (all with a `pos` column)

suppressPackageStartupMessages({
    library(arrow)
})

# ------------------------------------------------------------------ arguments
argv <- commandArgs(trailingOnly = TRUE)
get_arg <- function(key, default = NULL, required = FALSE) {
    i <- which(argv == paste0("--", key))
    if (length(i) == 1 && i < length(argv)) return(argv[i + 1])
    if (required) stop("missing required argument --", key)
    default
}

mode         <- get_arg("mode", required = TRUE)
task         <- get_arg("task", required = TRUE)
input_matrix <- get_arg("input_matrix", required = TRUE)
target_matrix<- get_arg("target_matrix", required = TRUE)
lists_dir    <- get_arg("lists_dir", required = TRUE)
out_dir      <- get_arg("out_dir", default = lists_dir)
tf_list      <- get_arg("tf_list", default = "../One_Shifting/data/DatabaseExtract_v_1.01.csv")
n_hvg        <- as.integer(get_arg("n_hvg", "1000"))
min_cells    <- as.integer(get_arg("min_cells", "3"))
test_frac    <- as.numeric(get_arg("test_frac", "0.2"))
val_frac     <- as.numeric(get_arg("val_frac", "0.1"))
seed         <- as.integer(get_arg("seed", "42"))
input_genes_override  <- get_arg("input_genes")
target_genes_override <- get_arg("target_genes")

if (!mode %in% c("lists", "extract")) stop("--mode must be lists or extract")
if (!task %in% c("within", "cross"))  stop("--task must be within or cross")
dir.create(lists_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir,   recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------ loading
ID_COLS <- c("pos", "index", "Unnamed: 0", "cell", "cell_id", "barcode")

read_matrix <- function(path, label) {
    if (!file.exists(path)) stop(label, " not found: ", path)
    df <- as.data.frame(read_feather(path))
    hit <- intersect(colnames(df), ID_COLS)
    if (length(hit) == 0) {
        stop(label, " (", path, ") has no cell-identifier column. One of {",
             paste(ID_COLS, collapse = ", "), "} is required: the two tasks pair cells ",
             "by name, and a positional pairing cannot be checked.")
    }
    ids <- as.character(df[[hit[1]]])
    df <- df[, !(colnames(df) %in% ID_COLS), drop = FALSE]
    if (anyDuplicated(ids) > 0) stop(label, " has duplicate cell ids")
    cat(sprintf("[INFO] %-6s %s : %d cells x %d genes\n", label, basename(path),
                nrow(df), ncol(df)))
    list(ids = ids, mat = df)
}

inp <- read_matrix(input_matrix, "input")
tgt <- if (normalizePath(input_matrix, mustWork = FALSE) ==
           normalizePath(target_matrix, mustWork = FALSE)) inp else
       read_matrix(target_matrix, "target")

# align by cell id
common_ids <- intersect(inp$ids, tgt$ids)
if (length(common_ids) == 0) stop("input and target share no cell ids")
if (length(common_ids) < length(inp$ids) || length(common_ids) < length(tgt$ids)) {
    cat(sprintf("[INFO] aligning on %d cells shared by both matrices (input had %d, target %d)\n",
                length(common_ids), length(inp$ids), length(tgt$ids)))
}
common_ids <- common_ids[order(match(common_ids, inp$ids))]   # keep input's order
X <- inp$mat[match(common_ids, inp$ids), , drop = FALSE]
Y <- tgt$mat[match(common_ids, tgt$ids), , drop = FALSE]
rownames(X) <- NULL; rownames(Y) <- NULL

split_path <- file.path(lists_dir, "cell_split.csv")

# ============================================================ mode: lists
if (mode == "lists") {
    suppressPackageStartupMessages(library(Seurat))

    # --- the split comes first, so nothing downstream can see the test cells ---
    set.seed(seed)
    n <- length(common_ids)
    n_test <- round(n * test_frac); n_val <- round(n * val_frac)
    if (n_test + n_val >= n) stop("test_frac + val_frac leave no training cells")
    perm <- sample(seq_len(n))
    test_i  <- sort(perm[seq_len(n_test)])
    val_i   <- sort(perm[n_test + seq_len(n_val)])
    train_i <- sort(perm[(n_test + n_val + 1):n])
    split_df <- data.frame(
        pos   = common_ids,
        split = ifelse(seq_len(n) %in% train_i, "train",
                ifelse(seq_len(n) %in% val_i,   "val", "test")),
        stringsAsFactors = FALSE)
    write.csv(split_df, split_path, row.names = FALSE)
    cat(sprintf("[INFO] split fixed: %d train / %d val / %d test  -> %s\n",
                length(train_i), length(val_i), length(test_i), split_path))

    # --- everything below sees TRAINING CELLS ONLY ---
    Xtr <- X[train_i, , drop = FALSE]
    Ytr <- Y[train_i, , drop = FALSE]

    nz_in  <- colSums(as.matrix(Xtr) != 0, na.rm = TRUE)
    nz_out <- colSums(as.matrix(Ytr) != 0, na.rm = TRUE)
    keep_in  <- colnames(Xtr)[nz_in  >= min_cells]
    keep_out <- colnames(Ytr)[nz_out >= min_cells]
    cat(sprintf("[INFO] training-cell filter (non-zero in >= %d cells): input %d/%d, target %d/%d\n",
                min_cells, length(keep_in), ncol(Xtr), length(keep_out), ncol(Ytr)))

    tf_db <- read.csv(tf_list, stringsAsFactors = FALSE)
    tf_all <- if ("HGNC.symbol" %in% colnames(tf_db)) {
        if ("Is.TF." %in% colnames(tf_db)) tf_db$HGNC.symbol[tf_db$Is.TF. == "Yes"]
        else tf_db$HGNC.symbol
    } else if ("Name" %in% colnames(tf_db)) tf_db$Name
      else if ("Gene" %in% colnames(tf_db)) tf_db$Gene
      else tf_db[[1]]
    tf_all <- unique(tf_all[tf_all != "" & !is.na(tf_all)])
    tf_in_input  <- intersect(keep_in,  tf_all)
    tf_in_target <- intersect(keep_out, tf_all)
    cat(sprintf("[INFO] TF annotation: %d symbols, %d present in filtered input, %d in filtered target\n",
                length(tf_all), length(tf_in_input), length(tf_in_target)))

    # HVGs of the TARGET matrix, training cells only
    sobj <- CreateSeuratObject(counts = t(as.matrix(Ytr[, keep_out, drop = FALSE])))
    sobj <- NormalizeData(sobj, verbose = FALSE)
    sobj <- FindVariableFeatures(sobj, selection.method = "vst",
                                 nfeatures = min(n_hvg, length(keep_out)), verbose = FALSE)
    hvg_all <- VariableFeatures(sobj)
    hvg_non_tf <- setdiff(hvg_all, tf_in_target)
    cat(sprintf("[INFO] HVGs on training cells: %d requested, %d returned, %d were TFs -> %d non-TF targets\n",
                n_hvg, length(hvg_all), length(intersect(hvg_all, tf_in_target)), length(hvg_non_tf)))

    if (task == "within") {
        input_genes <- tf_in_input
        # a gene must not be both the question and the answer
        target_genes <- setdiff(hvg_non_tf, input_genes)
        hvg_written <- hvg_non_tf
    } else {
        # cross predicts the RNA expression of the SAME genes whose ATAC activity it is
        # given, so there is one list, not two. Subsetting hvg_non_tf keeps HVG order;
        # intersect() would too, but this says so without relying on it. Genes present in
        # only one modality drop out here and cannot reach either matrix.
        target_genes <- hvg_non_tf[hvg_non_tf %in% keep_in]
        input_genes  <- target_genes
        hvg_written  <- target_genes
        cat(sprintf("[INFO] shared gene set: %d non-TF RNA HVG candidates, %d of them also pass the ATAC filter -> %d genes used for BOTH input and target\n",
                    length(hvg_non_tf), length(target_genes), length(target_genes)))
        if (length(target_genes) == 0) {
            stop("the non-TF RNA HVGs and the filtered ATAC features do not overlap, so ",
                 "there is no gene whose ATAC activity could predict its own RNA. ",
                 "Check that the two matrices use the same gene symbols; --n_hvg is a ",
                 "candidate count and is deliberately not padded to reach a target size.")
        }
    }
    if (length(input_genes) == 0)  stop("no input genes survive selection")
    if (length(target_genes) == 0) stop("no target genes survive selection")

    writeLines(tf_in_input,  file.path(lists_dir, "tf_names.txt"))
    writeLines(hvg_written,  file.path(lists_dir, "hvg_target_names.txt"))
    writeLines(input_genes,  file.path(lists_dir, "input_genes.txt"))
    writeLines(target_genes, file.path(lists_dir, "target_genes.txt"))
    cat(sprintf("[INFO] gene lists written to %s : %d input, %d target\n",
                lists_dir, length(input_genes), length(target_genes)))

    writeLines(sprintf(paste0(
        '{\n  "task": "%s",\n  "seed": %d,\n  "test_frac": %s,\n  "val_frac": %s,\n',
        '  "n_hvg": %d,\n  "min_cells": %d,\n  "n_cells_aligned": %d,\n',
        '  "n_cells_train": %d,\n  "n_cells_val": %d,\n  "n_cells_test": %d,\n',
        '  "n_hvg_candidates": %d,\n  "n_shared_genes": %s,\n',
        '  "n_input_genes": %d,\n  "n_target_genes": %d,\n',
        '  "shared_gene_set": %s,\n',
        '  "input_matrix": "%s",\n  "target_matrix": "%s"\n}\n'),
        task, seed, format(test_frac), format(val_frac), n_hvg, min_cells,
        n, length(train_i), length(val_i), length(test_i),
        length(hvg_non_tf),
        if (task == "cross") as.character(length(target_genes)) else "null",
        length(input_genes), length(target_genes),
        if (task == "cross") "true" else "false",
        input_matrix, target_matrix),
        file.path(lists_dir, "prep_meta.json"))
    quit(save = "no", status = 0)
}

# ============================================================ mode: extract
if (!file.exists(split_path)) {
    stop("no cell_split.csv in ", lists_dir, " -- run this script with --mode lists first")
}
split_df <- read.csv(split_path, stringsAsFactors = FALSE)
miss <- setdiff(split_df$pos, common_ids)
if (length(miss) > 0) {
    stop(length(miss), " cells in the fixed split are absent from these matrices (e.g. ",
         paste(utils::head(miss, 3), collapse = ", "), "). The split is per experiment; ",
         "a normalized matrix must contain the same cells as the matrix it was built from.")
}

read_list <- function(override, default_name) {
    p <- if (!is.null(override)) override else file.path(lists_dir, default_name)
    if (!file.exists(p)) stop("gene list not found: ", p)
    g <- trimws(readLines(p)); g[nzchar(g)]
}
input_genes  <- read_list(input_genes_override,  "input_genes.txt")
target_genes <- read_list(target_genes_override, "target_genes.txt")
if (task == "cross" && !identical(input_genes, target_genes)) {
    stop("for --task cross the input and target gene lists must be the same list in the ",
         "same order (", length(input_genes), " vs ", length(target_genes), " entries). ",
         "Re-run --mode lists into a fresh --lists_dir; these files were written by the ",
         "earlier two-list version of this script.")
}

miss_in  <- setdiff(input_genes,  colnames(X))
miss_out <- setdiff(target_genes, colnames(Y))
if (length(miss_in) > 0)  stop(length(miss_in),  " input genes from the fixed list are absent here (e.g. ",
                               paste(utils::head(miss_in, 3), collapse = ", "), ")")
if (length(miss_out) > 0) stop(length(miss_out), " target genes from the fixed list are absent here (e.g. ",
                               paste(utils::head(miss_out, 3), collapse = ", "), ")")

idx_of <- match(split_df$pos, common_ids)
for (part in c("train", "val", "test")) {
    rows <- idx_of[split_df$split == part]
    ids  <- common_ids[rows]
    xi <- X[rows, input_genes,  drop = FALSE]; xi <- cbind(pos = ids, xi)
    yi <- Y[rows, target_genes, drop = FALSE]; yi <- cbind(pos = ids, yi)
    write_feather(xi, file.path(out_dir, sprintf("input_%s.feather",  part)))
    write_feather(yi, file.path(out_dir, sprintf("target_%s.feather", part)))
    cat(sprintf("[INFO] %-5s : input %d x %d, target %d x %d\n", part,
                nrow(xi), ncol(xi) - 1, nrow(yi), ncol(yi) - 1))
}
cat(sprintf("[DONE] matrices written to %s\n", out_dir))