load_UMAP_cbmc <- function(save_dir) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(SeuratData)
        library(Seurat)
        library(arrow)
        library(tibble)
    })

    dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

    InstallData("cbmc")
    data("cbmc")
    cbmc <- UpdateSeuratObject(cbmc)
    colnames(cbmc@meta.data)

    table(cbmc$rna_annotations)
    table(cbmc$protein_annotations)
    table(Idents(cbmc))
    head(cbmc@meta.data, 10)

    # filter - use protein_annotations
    cbmc_clean <- subset(cbmc, protein_annotations != "Mouse" & 
                            protein_annotations != "Multiplets" & 
                            protein_annotations != "T/Mono doublets")

    # ===== Raw counts =====
    rna_counts <- as.matrix(GetAssayData(cbmc_clean, assay = "RNA", layer = "counts"))
    dim(rna_counts)
    rna_counts[1:5, 1:5]
    rna_df <- as.data.frame(rna_counts)
    rna_df <- rownames_to_column(rna_df, var = "pos")
    rna_df[1:5,1:5]
    dim(rna_df)

    data_cols <- rna_df[, -1]

    nonzero_rows <- rowSums(data_cols) > 0
    nonzero_cols <- colSums(data_cols) > 0

    cat("All-zero genes:", sum(!nonzero_rows), "\n")
    cat("All-zero cells:", sum(!nonzero_cols), "\n")
    rna_df <- rna_df[nonzero_rows, c(TRUE, nonzero_cols)]
    cat("After filtering:", dim(rna_df)[1], "genes x", dim(rna_df)[2]-1, "cells\n")

    # extract cell type - use protein_annotations
    cell_types <- cbmc_clean$protein_annotations
    table(cell_types)

    celltype_df <- data.frame(
        barcode = colnames(cbmc_clean),
        cell_type = cbmc_clean$protein_annotations
    )

    dim(rna_counts)
    length(cell_types)

    # Save raw counts (genes x cells)
    write_feather(rna_df, file.path(save_dir, "orig_data/cbmc_rna_counts.feather"))
    write.csv(rna_df, file.path(save_dir, "orig_data/cbmc_rna_counts.csv"), row.names = TRUE)  
    write.csv(celltype_df, file.path(save_dir, "orig_data/cbmc_celltype_protein.csv"), row.names = TRUE)  

    # ===== Seurat standard normalization (LogNormalize, scale.factor = 10000) =====
    cbmc_clean <- NormalizeData(cbmc_clean, normalization.method = "LogNormalize", scale.factor = 10000)
    
    rna_norm <- as.matrix(GetAssayData(cbmc_clean, assay = "RNA", layer = "data"))
    
    # apply same gene/cell filter
    keep_genes <- rownames(rna_norm) %in% rna_df$pos
    keep_cells <- colnames(rna_norm) %in% colnames(rna_df)[-1]
    rna_norm <- rna_norm[keep_genes, keep_cells]
    
    cat("Normalized matrix:", nrow(rna_norm), "genes x", ncol(rna_norm), "cells\n")

    # genes x cells (same orientation as raw)
    norm_df <- as.data.frame(rna_norm)
    norm_df <- rownames_to_column(norm_df, var = "pos")
    write_feather(norm_df, file.path(save_dir, "orig_data/cbmc_rna_seurat_norm.feather"))
    cat("Saved: cbmc_rna_seurat_norm.feather (genes x cells)\n")

    # cells x genes (transposed)
    norm_t <- t(rna_norm)
    norm_t_df <- as.data.frame(norm_t)
    norm_t_df <- rownames_to_column(norm_t_df, var = "pos")
    write_feather(norm_t_df, file.path(save_dir, "orig_data/cbmc_rna_seurat_norm_transposed.feather"))
    cat("Saved: cbmc_rna_seurat_norm_transposed.feather (cells x genes)\n")
}
