load_save_seurat_pbmc <- function(save_dir, keep_perc = NULL) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(SeuratData)
        library(Seurat)
        library(Signac)
        library(EnsDb.Hsapiens.v86)
        library(GenomicRanges)
        library(dplyr)
        library(arrow)  
        library(tibble)
    })

    save_sample_hist <- function(mat, out_file, title, sample_frac = 0.10, max_sample = 1e6, x_max = 40) {
        nr <- nrow(mat)
        nc <- ncol(mat)
        total_n <- nr * nc

        sample_size <- min(ceiling(total_n * sample_frac), max_sample)

        set.seed(123)
        sample_idx <- sample.int(total_n, size = sample_size, replace = FALSE)

        sample_rows <- ((sample_idx - 1) %% nr) + 1
        sample_cols <- ((sample_idx - 1) %/% nr) + 1

        sampled_values <- as.numeric(mat[cbind(sample_rows, sample_cols)])
        sampled_values <- sampled_values[is.finite(sampled_values)]

        png(out_file, width = 1200, height = 800, res = 150)

        hist(
            sampled_values[sampled_values <= x_max],
            breaks = seq(-0.5, x_max + 0.5, by = 1),
            probability = TRUE,
            main = paste0(title, "\n10% sampled entries including zeros; n = ", sample_size),
            xlab = "Count",
            ylab = "Proportion"
        )

        dev.off()

        cat("Saved histogram:", out_file, "\n")
        cat("Sampled zero proportion:", mean(sampled_values == 0), "\n")
    }

    out_dir <- save_dir
    if (!dir.exists(out_dir)) {
        dir.create(out_dir, recursive = TRUE)
    }
    path <- paste0(out_dir, "/ATAC")
    if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
    }
    path <- paste0(out_dir, "/RNA")
    if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
    }

    pbmc.rna <- LoadData("pbmcMultiome", "pbmc.rna")
    pbmc.atac <- LoadData("pbmcMultiome", "pbmc.atac")

    # === INFO: Unfiltered dataset dimensions ===
    cat("INFO [unfiltered] RNA cells:", ncol(pbmc.rna), "\n")
    cat("INFO [unfiltered] RNA genes:", nrow(pbmc.rna), "\n")
    cat("INFO [unfiltered] ATAC cells:", ncol(pbmc.atac), "\n")
    cat("INFO [unfiltered] ATAC peaks:", nrow(pbmc.atac), "\n")

    pbmc.rna[["RNA"]] <- as(pbmc.rna[["RNA"]], Class = "Assay5")
    pbmc.rna <- subset(pbmc.rna, seurat_annotations != "filtered")
    pbmc.atac <- subset(pbmc.atac, seurat_annotations != "filtered")

    # === INFO: Cells retained after QC filtering ===
    cat("INFO [after QC] RNA cells retained:", ncol(pbmc.rna), "\n")
    cat("INFO [after QC] ATAC cells retained:", ncol(pbmc.atac), "\n")
    cat("INFO [after QC] RNA genes retained:", nrow(pbmc.rna), "\n")
    cat("INFO [after QC] ATAC peaks retained:", nrow(pbmc.atac), "\n")

    annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
    seqlevelsStyle(annotations) <- "UCSC"
    genome(annotations) <- "hg38"
    Annotation(pbmc.atac) <- annotations

    gene_annot <- annotations # Number of genes: 28465
    length(unique(gene_annot$gene_name)) # 28465

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

    upstream <- 2000
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

    gene_lengths <- width(gene_coords)
    names(gene_lengths) <- gene_coords$gene_name

    # peaks <- granges(pbmc.atac)
    peaks <- granges(pbmc.atac[["ATAC"]])
    cat("Number of peaks:", length(peaks), "\n")

    overlaps <- findOverlaps(peaks, gene_coords)
    cat("Number of overlaps:", length(overlaps), "\n")

    atac_counts <- GetAssayData(pbmc.atac, assay = "ATAC", layer = "counts")

    gene_names <- gene_coords$gene_name
    gene.activities <- matrix(0, 
                            nrow = length(gene_coords), 
                            ncol = ncol(atac_counts),
                            dimnames = list(gene_names, colnames(atac_counts)))

    for (i in seq_along(gene_coords)) {
        peak_idx <- queryHits(overlaps)[subjectHits(overlaps) == i]
        if (length(peak_idx) > 0) {
            if (length(peak_idx) == 1) {
            gene.activities[i, ] <- atac_counts[peak_idx, ]
            } else {
            gene.activities[i, ] <- Matrix::colSums(atac_counts[peak_idx, , drop = FALSE])
            }
        }
    }

    non_zero_mask <- rowSums(gene.activities) > 0
    gene.activities <- gene.activities[non_zero_mask, ]
    cat("Genes with activity (after filtering):", nrow(gene.activities), "\n")

    gene.activities <- as(gene.activities, "sparseMatrix")
    cat("Final dimensions:", dim(gene.activities), "\n")

    pbmc.atac[["ACTIVITY"]] <- CreateAssayObject(counts = gene.activities)

    all.equal(colnames(pbmc.atac), colnames(pbmc.rna))

    # === corr ===
    common_genes <- intersect(rownames(pbmc.atac[["ACTIVITY"]]), rownames(pbmc.rna[["RNA"]]))
    cat("Common genes (initial):", length(common_genes), "\n")

    # extract matrix with shared genes
    activity_mat <- as.matrix(GetAssayData(pbmc.atac, assay = "ACTIVITY", layer = "counts")[common_genes, ])
    rna_mat <- as.matrix(GetAssayData(pbmc.rna, assay = "RNA", layer = "counts")[common_genes, ])

    cat("Activity dimensions (before filtering):", dim(activity_mat), "\n")
    cat("RNA dimensions (before filtering):", dim(rna_mat), "\n")

    # Remove all 0 rows
    activity_non_zero <- rowSums(activity_mat) > 0
    rna_non_zero <- rowSums(rna_mat) > 0

    cat("Activity non-zero genes:", sum(activity_non_zero), "\n")
    cat("RNA non-zero genes:", sum(rna_non_zero), "\n")

    keep_genes <- activity_non_zero & rna_non_zero
    cat("Genes to keep (non-zero in both):", sum(keep_genes), "\n")

    # Filter Matrix
    activity_mat <- activity_mat[keep_genes, ]
    rna_mat <- rna_mat[keep_genes, ]

    cat("Activity dimensions (after filtering):", dim(activity_mat), "\n")
    cat("RNA dimensions (after filtering):", dim(rna_mat), "\n")

    ## cell corr
    temp_cell_cor <- sapply(1:ncol(activity_mat), function(x) {
        cor(activity_mat[, x], rna_mat[, x])
    })
    names(temp_cell_cor) <- colnames(activity_mat)

    cat("Cell correlation - Mean:", mean(temp_cell_cor, na.rm = TRUE), "\n")
    cat("Cell correlation - Median:", median(temp_cell_cor, na.rm = TRUE), "\n")

    ## gene corr
    temp_gene_cor <- sapply(1:nrow(activity_mat), function(x) {
        cor(activity_mat[x, ], rna_mat[x, ])
    })
    names(temp_gene_cor) <- rownames(activity_mat)

    cat("Gene correlation - Mean:", mean(temp_gene_cor, na.rm = TRUE), "\n")
    cat("Gene correlation - Median:", median(temp_gene_cor, na.rm = TRUE), "\n")

    # === save to feather ====
    # ==== save raw counts (not transposed) ====
    filtered_gene_names <- rownames(activity_mat)

    activity_counts <- as.matrix(GetAssayData(pbmc.atac, assay = "ACTIVITY", layer = "counts")[filtered_gene_names, ])
    rna_counts <- as.matrix(GetAssayData(pbmc.rna, assay = "RNA", layer = "counts")[filtered_gene_names, ])

    # === INFO: Sparsity of final count matrices ===
    cat("INFO [final] Activity counts dimensions:", dim(activity_counts), "\n")
    cat("INFO [final] RNA counts dimensions:", dim(rna_counts), "\n")

    activity_total_elements <- prod(dim(activity_counts))
    activity_zero_elements <- sum(activity_counts == 0)
    activity_sparsity <- activity_zero_elements / activity_total_elements * 100
    cat("INFO [final] Activity (ATAC) sparsity:", round(activity_sparsity, 2), "%\n")

    rna_total_elements <- prod(dim(rna_counts))
    rna_zero_elements <- sum(rna_counts == 0)
    rna_sparsity <- rna_zero_elements / rna_total_elements * 100
    cat("INFO [final] RNA sparsity:", round(rna_sparsity, 2), "%\n")

    # Save raw counts
    dir.create(file.path(out_dir, "PBMC_RNA"), recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(out_dir, "PBMC_ATAC"), recursive = TRUE, showWarnings = FALSE)

    # Activity raw counts: gene x cell
    activity_counts_df <- as.data.frame(activity_counts)
    activity_counts_df <- rownames_to_column(activity_counts_df, var = "pos") 
    activity_counts_df[1:5,1:5]
    write_feather(activity_counts_df, paste0(out_dir, "/PBMC_ATAC/activity_counts.feather"))
    save_sample_hist(
        atac_counts,
        paste0(out_dir, "/PBMC_ATAC/atac_peak_counts_hist.png"),
        "PBMC ATAC peak-level counts histogram (10% sample)"
    )
    save_sample_hist(
        activity_counts,
        paste0(out_dir, "/PBMC_ATAC/activity_counts_hist.png"),
        "PBMC ATAC activity histogram (10% sample)"
    )

    # RNA raw counts: gene x cell
    rna_counts_df <- as.data.frame(rna_counts)
    rna_counts_df <- rownames_to_column(rna_counts_df, var = "pos") 
    rna_counts_df[1:5,1:5]
    write_feather(rna_counts_df, paste0(out_dir, "/PBMC_RNA/rna_counts.feather"))
    save_sample_hist(
        rna_counts,
        paste0(out_dir, "/PBMC_RNA/rna_counts_hist.png"),
        "PBMC RNA counts histogram (10% sample)"
    )

    cat("  - activity_counts.feather\n")
    cat("  - rna_counts.feather\n")
    cat("  - atac_peak_counts_hist.png\n")
    cat("  - activity_counts_hist.png\n")
    cat("  - rna_counts_hist.png\n")

    return(list(rna_counts = rna_counts, activity_counts_df = activity_counts_df))
}


args <- commandArgs(trailingOnly = TRUE)

save_dir <- args[1]
load_save_seurat_pbmc(save_dir = save_dir, keep_perc = NULL)
