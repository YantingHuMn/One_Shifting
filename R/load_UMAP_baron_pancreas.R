load_UMAP_baron_pancreas <- function(save_dir) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(Seurat)
        library(tibble)
        library(arrow)
    })

    dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
    raw_dir <- file.path(save_dir, "raw_download")
    dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

    # download RAW tar
    tar_url <- "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84133/suppl/GSE84133_RAW.tar"
    tar_file <- file.path(raw_dir, "GSE84133_RAW.tar")

    if (!file.exists(tar_file)) {
        cat("Downloading Baron pancreas RAW data...\n")
        download.file(tar_url, tar_file, mode = "wb")
    }

    # unzip
    untar(tar_file, exdir = raw_dir)
    cat("Extracted files:\n")
    extracted <- list.files(raw_dir, pattern = "\\.csv\\.gz$")
    print(extracted)

    # only read human
    human_files <- extracted[grepl("[Hh]uman", extracted)]
    cat("Human files:", human_files, "\n")

    if (length(human_files) == 0) {
        cat("No 'human' in filenames. All files:\n")
        print(extracted)
        cat("Reading first file to inspect...\n")
        test <- read.csv(gzfile(file.path(raw_dir, extracted[1])), nrows = 5, check.names = FALSE)
        print(head(test[, 1:5]))
        stop("Please check file names and adjust the script.")
    }

    all_data <- list()
    for (f in human_files) {
        cat("Reading:", f, "\n")
        dat <- read.csv(gzfile(file.path(raw_dir, f)), row.names = 1, check.names = FALSE)
        cat("  Dimensions:", dim(dat), "\n")
        cat("  First 3 colnames:", head(colnames(dat), 3), "\n")
        all_data[[f]] <- dat
    }

    # combine（row=cell）
    combined <- do.call(rbind, all_data)
    cat("Combined:", dim(combined), "\n")
    cat("First 3:", combined[1:3,1:3], "\n")

    # Baron format: row=cell, first col=assigned_cluster, other=gene
    cell_types <- combined[, "assigned_cluster"]
    meta_cols <- c("barcode", "assigned_cluster")
    gene_cols <- setdiff(colnames(combined), meta_cols)
    counts_t <- combined[, gene_cols]

    # transpose to genes x cells
    counts <- t(as.matrix(counts_t))
    cat("Count matrix:", nrow(counts), "genes x", ncol(counts), "cells\n")

    # filter all zero genes and cells
    rna_df <- as.data.frame(counts)
    rna_df <- rownames_to_column(rna_df, var = "pos")

    data_cols <- rna_df[, -1]
    nonzero_rows <- rowSums(data_cols) > 0
    nonzero_cols <- colSums(data_cols) > 0

    cat("All-zero genes:", sum(!nonzero_rows), "\n")
    cat("All-zero cells:", sum(!nonzero_cols), "\n")
    rna_df <- rna_df[nonzero_rows, c(TRUE, nonzero_cols)]
    cat("After filtering:", dim(rna_df)[1], "genes x", dim(rna_df)[2] - 1, "cells\n")

    keep_cells <- nonzero_cols
    cell_types_filtered <- cell_types[keep_cells]

    celltype_df <- data.frame(
        barcode = colnames(rna_df)[-1],
        cell_type = cell_types_filtered
    )

    write_feather(rna_df, file.path(save_dir, "orig_data/baron_pancreas_rna_counts.feather"))
    write.csv(rna_df, file.path(save_dir, "orig_data/baron_pancreas_rna_counts.csv"), row.names = FALSE)
    write.csv(celltype_df, file.path(save_dir, "orig_data/baron_pancreas_celltype.csv"), row.names = FALSE)

    # Seurat normalize
    norm_mat <- as.matrix(rna_df[, -1])
    rownames(norm_mat) <- rna_df$pos

    seurat_tmp <- CreateSeuratObject(counts = norm_mat)
    seurat_tmp <- NormalizeData(seurat_tmp, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)

    norm_data <- as.matrix(GetAssayData(seurat_tmp, assay = "RNA", layer = "data"))

    norm_df <- as.data.frame(norm_data)
    norm_df <- rownames_to_column(norm_df, var = "pos")
    write_feather(norm_df, file.path(save_dir, "orig_data/baron_pancreas_rna_seurat_norm.feather"))

    norm_df_t <- as.data.frame(t(norm_data))
    norm_df_t <- rownames_to_column(norm_df_t, var = "pos")
    write_feather(norm_df_t, file.path(save_dir, "orig_data/baron_pancreas_rna_seurat_norm_transposed.feather"))

    cat("Saved all files to:", save_dir, "\n")
}

