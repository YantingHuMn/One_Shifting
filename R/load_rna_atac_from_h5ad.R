load_rna_atac_from_h5ad <- function(path, sample_name, out_dir, upstream = 2000) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(hdf5r)
        library(Matrix)
        library(Signac)
        library(EnsDb.Hsapiens.v86)
        library(GenomicRanges)
        library(dplyr)
        library(anndata)
        library(arrow)
    })

    adata <- read_h5ad(path)
    mat <- adata$X
    mat <- t(mat)
    mat <- as(mat, "sparseMatrix")
    rownames(mat) <- adata$var_names
    colnames(mat) <- adata$obs_names

    feature_type <- adata$var$feature_types
    rna_idx <- which(feature_type == "Gene Expression")
    atac_idx <- which(feature_type == "Chromatin Accessibility")

    rna_mat <- mat[rna_idx, ]
    atac_mat <- mat[atac_idx, ]

    cat("RNA dim:", dim(rna_mat), "\n")
    cat("ATAC dim:", dim(atac_mat), "\n")

    # Remove ENSG rows
    ensg_indices <- grep("^ENSG", rownames(rna_mat))
    if (length(ensg_indices) > 0) {
        ensg_ids <- sub("\\..*", "", rownames(rna_mat)[ensg_indices])
        gene_map <- ensembldb::select(EnsDb.Hsapiens.v86,
                                        keys = ensg_ids,
                                        keytype = "GENEID",
                                        columns = c("GENEID", "SYMBOL"))
        id_to_symbol <- setNames(gene_map$SYMBOL, gene_map$GENEID)
        new_names <- id_to_symbol[sub("\\..*", "", rownames(rna_mat))]

        mapped <- !is.na(new_names) & !duplicated(new_names)
        rna_mat_cleaned <- rna_mat[mapped, ]
        rownames(rna_mat_cleaned) <- new_names[mapped]
    } else {
        rna_mat_cleaned <- rna_mat
    }
    cat("RNA after ENSG conversion:", dim(rna_mat_cleaned), "\n")

    # Standardize underscores to dashes
    rownames(rna_mat_cleaned) <- gsub("_", "-", rownames(rna_mat_cleaned))

    # Peak coordinates
    peak_names <- rownames(atac_mat)
    peak_df <- do.call(rbind, strsplit(peak_names, "[:-]"))
    peak_gr <- GRanges(
        seqnames = peak_df[, 1],
        ranges = IRanges(start = as.numeric(peak_df[, 2]), end = as.numeric(peak_df[, 3]))
    )
    names(peak_gr) <- peak_names

    # Gene annotation
    annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
    seqlevelsStyle(annotations) <- "UCSC"
    genome(annotations) <- "hg38"

    gene_annot <- annotations

    gene_df <- data.frame(
        chr = as.character(seqnames(gene_annot)),
        start = start(gene_annot),
        end = end(gene_annot),
        strand = as.character(strand(gene_annot)),
        gene_name = gene_annot$gene_name
    )

    gene_summary <- gene_df %>%
        group_by(gene_name) %>%
        summarise(
            chr = names(sort(table(chr), decreasing = TRUE))[1],
            start = min(start),
            end = max(end),
            strand = names(sort(table(strand), decreasing = TRUE))[1],
            .groups = "drop"
        )

    gene_coords <- GRanges(
        seqnames = gene_summary$chr,
        ranges = IRanges(
            start = ifelse(gene_summary$strand == "+",
                           gene_summary$start - upstream,
                           gene_summary$start),
            end = ifelse(gene_summary$strand == "-",
                         gene_summary$end + upstream,
                         gene_summary$end)
        ),
        strand = gene_summary$strand,
        gene_name = gene_summary$gene_name
    )

    start(gene_coords) <- pmax(start(gene_coords), 1)
    gene_coords <- keepStandardChromosomes(gene_coords, pruning.mode = "coarse")
    cat("Number of genes:", length(gene_coords), "\n")

    # Compute overlaps & gene activity
    overlaps <- findOverlaps(peak_gr, gene_coords)
    cat("Number of overlaps:", length(overlaps), "\n")

    gene_names <- gene_coords$gene_name
    gene.activities <- matrix(0,
                              nrow = length(gene_coords),
                              ncol = ncol(atac_mat),
                              dimnames = list(gene_names, colnames(atac_mat)))

    for (i in seq_along(gene_coords)) {
        peak_idx <- queryHits(overlaps)[subjectHits(overlaps) == i]
        if (length(peak_idx) > 0) {
            if (length(peak_idx) == 1) {
                gene.activities[i, ] <- atac_mat[peak_idx, ]
            } else {
                gene.activities[i, ] <- Matrix::colSums(atac_mat[peak_idx, , drop = FALSE])
            }
        }
    }

    non_zero_mask <- rowSums(gene.activities) > 0
    gene.activities <- gene.activities[non_zero_mask, ]
    cat("Genes with activity:", nrow(gene.activities), "\n")

    # Standardize underscores to dashes
    rownames(gene.activities) <- gsub("_", "-", rownames(gene.activities))

    # Common genes and shared cells
    common_genes <- intersect(rownames(gene.activities), rownames(rna_mat_cleaned))
    cat("Common genes:", length(common_genes), "\n")

    if (length(common_genes) == 0) stop("No common genes found! Check rowname formats above.")

    rna_counts <- as.matrix(rna_mat_cleaned[common_genes, ])
    activity_counts <- as.matrix(gene.activities[common_genes, ])

    shared_cells <- intersect(colnames(rna_counts), colnames(activity_counts))
    rna_counts <- rna_counts[, shared_cells]
    activity_counts <- activity_counts[, shared_cells]

    cat("Aligned matrix dim:", dim(rna_counts), "\n")

    # Correlation (on raw counts, just for diagnostics)
    cell_cor <- sapply(1:ncol(activity_counts), function(x) cor(activity_counts[, x], rna_counts[, x]))
    cat("Cell correlation - Mean:", mean(cell_cor, na.rm = TRUE), "\n")
    cat("Cell correlation - Median:", median(cell_cor, na.rm = TRUE), "\n")

    gene_cor <- sapply(1:nrow(activity_counts), function(x) cor(activity_counts[x, ], rna_counts[x, ]))
    cat("Gene correlation - Mean:", mean(gene_cor, na.rm = TRUE), "\n")
    cat("Gene correlation - Median:", median(gene_cor, na.rm = TRUE), "\n")

    # Save raw counts
    dir.create(file.path(out_dir, paste0(sample_name, "_RNA")), recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(out_dir, paste0(sample_name, "_ATAC")), recursive = TRUE, showWarnings = FALSE)

    # Filter to keep only genes non-zero in both
    keep_genes <- rowSums(rna_counts) > 0 & rowSums(activity_counts) > 0
    rna_counts <- rna_counts[keep_genes, ]
    activity_counts <- activity_counts[keep_genes, ]
    cat("Genes kept (non-zero in both):", sum(keep_genes), "\n")

    # Save as gene x cell with pos column
    rna_counts_df <- as.data.frame(rna_counts)
    rna_counts_df <- cbind(pos = rownames(rna_counts), rna_counts_df)
    rownames(rna_counts_df) <- NULL

    activity_counts_df <- as.data.frame(activity_counts)
    activity_counts_df <- cbind(pos = rownames(activity_counts), activity_counts_df)
    rownames(activity_counts_df) <- NULL

    write_feather(rna_counts_df, file.path(out_dir, paste0(sample_name, "_RNA"), "rna_counts.feather"))
    write_feather(activity_counts_df, file.path(out_dir, paste0(sample_name, "_ATAC"), "activity_counts.feather"))

    cat("  - rna_counts.feather\n")
    cat("  - activity_counts.feather\n")
    cat("Files saved to:", out_dir, "\n")
}

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
sample_name <- args[2]
out_dir <- args[3]

load_rna_atac_from_h5ad(path, sample_name, out_dir)