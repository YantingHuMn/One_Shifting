load_UMAP_zheng_duo8 <- function(save_dir, use_hvg = FALSE, n_hvg = 2000) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(DuoClustering2018)
        library(Seurat)
        library(tibble)
        library(arrow)
    })

    dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

    orig_data_dir <- file.path(save_dir, "orig_data")
    dir.create(orig_data_dir, recursive = TRUE, showWarnings = FALSE)

    # load data
    sce <- sce_full_Zhengmix8eq()

    print(sce)
    colnames(colData(sce))
    table(colData(sce)$phenoid)

    # convert to Seurat
    zheng <- as.Seurat(sce, counts = "counts", data = NULL)
    zheng <- UpdateSeuratObject(zheng)

    colnames(zheng@meta.data)
    head(zheng@meta.data, 10)

    zheng_clean <- zheng

    rna_counts <- as.matrix(
        GetAssayData(
            zheng_clean,
            assay = "originalexp",
            layer = "counts"
        )
    )

    dim(rna_counts)
    rna_counts[1:5, 1:5]

    rna_df <- as.data.frame(rna_counts)
    rna_df <- rownames_to_column(rna_df, var = "pos")

    rna_df[1:5, 1:5]
    dim(rna_df)

    data_cols <- rna_df[, -1]

    nonzero_rows <- rowSums(data_cols) > 0
    nonzero_cols <- colSums(data_cols) > 0

    cat("All-zero genes:", sum(!nonzero_rows), "\n")
    cat("All-zero cells:", sum(!nonzero_cols), "\n")

    rna_df <- rna_df[nonzero_rows, c(TRUE, nonzero_cols)]

    cat(
        "After filtering:",
        dim(rna_df)[1],
        "genes x",
        dim(rna_df)[2] - 1,
        "cells\n"
    )

    # Optional HVG filtering
    if (use_hvg) {
        cat("Selecting HVGs with n_hvg =", n_hvg, "\n")

        retained_cells <- colnames(rna_df)[-1]

        zheng_hvg <- subset(
            zheng_clean,
            cells = retained_cells
        )

        DefaultAssay(zheng_hvg) <- "originalexp"

        zheng_hvg <- NormalizeData(
            zheng_hvg,
            assay = "originalexp",
            normalization.method = "LogNormalize",
            scale.factor = 10000,
            verbose = FALSE
        )

        zheng_hvg <- FindVariableFeatures(
            zheng_hvg,
            assay = "originalexp",
            selection.method = "vst",
            nfeatures = n_hvg,
            verbose = FALSE
        )

        hvg_genes <- VariableFeatures(
            zheng_hvg[["originalexp"]]
        )

        hvg_genes <- intersect(
            hvg_genes,
            rna_df$pos
        )

        cat("Selected HVGs:", length(hvg_genes), "\n")

        rna_df <- rna_df[
            rna_df$pos %in% hvg_genes,
            ,
            drop = FALSE
        ]

        cat(
            "After HVG filtering:",
            dim(rna_df)[1],
            "genes x",
            dim(rna_df)[2] - 1,
            "cells\n"
        )
    }

    # FACS sorting labels
    cell_types <- colData(sce)$phenoid
    table(cell_types)

    celltype_df <- data.frame(
        barcode = colnames(zheng_clean),
        cell_type = cell_types
    )

    # Keep only cells retained after all-zero-cell filtering
    celltype_df <- celltype_df[
        celltype_df$barcode %in% colnames(rna_df)[-1],
        ,
        drop = FALSE
    ]

    write_feather(
        rna_df,
        file.path(orig_data_dir, "zheng_pbmc_rna_counts.feather")
    )

    write.csv(
        rna_df,
        file.path(orig_data_dir, "zheng_pbmc_rna_counts.csv"),
        row.names = FALSE
    )

    write.csv(
        celltype_df,
        file.path(orig_data_dir, "zheng_pbmc_celltype_facs.csv"),
        row.names = FALSE
    )

    # Seurat normalize
    norm_mat <- as.matrix(rna_df[, -1])
    rownames(norm_mat) <- rna_df$pos

    seurat_tmp <- CreateSeuratObject(counts = norm_mat)

    seurat_tmp <- NormalizeData(
        seurat_tmp,
        normalization.method = "LogNormalize",
        scale.factor = 10000,
        verbose = FALSE
    )

    norm_data <- as.matrix(
        GetAssayData(
            seurat_tmp,
            assay = "RNA",
            layer = "data"
        )
    )

    seurat_hvg_2000 <- FindVariableFeatures(
        seurat_tmp,
        selection.method = "vst",
        nfeatures = 2000,
        verbose = FALSE
    )

    hvg_2000_genes <- VariableFeatures(seurat_hvg_2000[["RNA"]])
    hvg_2000_genes <- intersect(
        hvg_2000_genes,
        rownames(norm_data)
    )

    writeLines(
        hvg_2000_genes,
        file.path(orig_data_dir, "hvg_2000.txt")
    )

    cat("Saved HVG 2000 list:", length(hvg_2000_genes), "\n")
    
    # genes x cells
    norm_df <- as.data.frame(norm_data)
    norm_df <- rownames_to_column(norm_df, var = "pos")

    write_feather(
        norm_df,
        file.path(orig_data_dir, "zheng_pbmc_rna_seurat_norm.feather")
    )

    # cells x genes (transposed)
    norm_df_t <- as.data.frame(t(norm_data))
    norm_df_t <- rownames_to_column(norm_df_t, var = "pos")

    write_feather(
        norm_df_t,
        file.path(
            orig_data_dir,
            "zheng_pbmc_rna_seurat_norm_transposed.feather"
        )
    )

    cat("Saved all files to:", save_dir, "\n")
}