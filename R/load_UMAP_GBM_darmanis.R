load_UMAP_GBM_darmanis <- function(save_dir) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(Seurat)
        library(tibble)
        library(arrow)
    })

    dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
    raw_dir <- file.path(save_dir, "raw_download")
    dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
    orig_dir <- file.path(save_dir, "orig_data")
    dir.create(orig_dir, recursive = TRUE, showWarnings = FALSE)

    expr_url <- "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84465/suppl/GSE84465_GBM_All_data.csv.gz"
    expr_file <- file.path(raw_dir, "GSE84465_GBM_All_data.csv.gz")

    if (!file.exists(expr_file)) {
        cat("Downloading GBM expression data...\n")
        download.file(expr_url, expr_file, mode = "wb")
    }

    cat("Reading expression matrix...\n")
    expr_data <- read.table(gzfile(expr_file), row.names = 1, header = TRUE, check.names = FALSE)
    cat("Dimensions:", nrow(expr_data), "genes x", ncol(expr_data), "cells\n")

    series_url <- "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84465/matrix/GSE84465_series_matrix.txt.gz"
    series_file <- file.path(raw_dir, "GSE84465_series_matrix.txt.gz")

    if (!file.exists(series_file)) {
        cat("Downloading GEO series matrix...\n")
        download.file(series_url, series_file, mode = "wb")
    }

    cat("Parsing series matrix for metadata...\n")
    all_lines <- readLines(gzfile(series_file))

    extract_field <- function(lines, field_name, occurrence = 1) {
        matches <- grep(paste0("^", field_name, "\t"), lines)
        if (length(matches) < occurrence) return(NULL)
        line <- lines[matches[occurrence]]
        values <- strsplit(line, "\t")[[1]][-1]
        values <- gsub('^"|"$', '', values)
        return(values)
    }

    clean_field <- function(x) gsub("^[^:]+:\\s*", "", x)

    gsm_ids     <- extract_field(all_lines, "!Sample_geo_accession")
    plate_ids   <- clean_field(extract_field(all_lines, "!Sample_characteristics_ch1", occurrence = 2))
    wells       <- clean_field(extract_field(all_lines, "!Sample_characteristics_ch1", occurrence = 3))
    cell_types  <- clean_field(extract_field(all_lines, "!Sample_characteristics_ch1", occurrence = 7))

    meta <- data.frame(
        gsm = gsm_ids,
        cell_barcode = paste0(plate_ids, ".", wells),
        cell_type = cell_types,
        stringsAsFactors = FALSE
    )

    cat("Cell types found:\n")
    print(table(meta$cell_type))

    # Pair
    common_cells <- intersect(colnames(expr_data), meta$cell_barcode)
    cat("Cells in expression matrix:", ncol(expr_data), "\n")
    cat("Cells in metadata:", nrow(meta), "\n")
    cat("Matched cells:", length(common_cells), "\n")

    if (length(common_cells) == 0) {
        cat("Expression matrix colnames (first 5):", paste(head(colnames(expr_data), 5), collapse = ", "), "\n")
        cat("Metadata barcodes (first 5):", paste(head(meta$cell_barcode, 5), collapse = ", "), "\n")
        stop("No matching cell barcodes!")
    }

    expr_data <- expr_data[, common_cells]
    rownames(meta) <- meta$cell_barcode
    meta <- meta[common_cells, ]

    # Filter all-zero genes and cells
    rna_counts <- as.matrix(expr_data)
    rna_df <- as.data.frame(rna_counts)
    rna_df <- rownames_to_column(rna_df, var = "pos")

    data_cols <- rna_df[, -1]
    nonzero_rows <- rowSums(data_cols) > 0
    nonzero_cols <- colSums(data_cols) > 0

    cat("All-zero genes:", sum(!nonzero_rows), "\n")
    cat("All-zero cells:", sum(!nonzero_cols), "\n")
    rna_df <- rna_df[nonzero_rows, c(TRUE, nonzero_cols)]
    cat("After filtering:", dim(rna_df)[1], "genes x", dim(rna_df)[2] - 1, "cells\n")

    # Cell type
    kept_cells <- colnames(rna_df)[-1]
    celltype_df <- meta[kept_cells, c("cell_barcode", "cell_type")]
    colnames(celltype_df) <- c("barcode", "cell_type")
    print(table(celltype_df$cell_type))

    write_feather(rna_df, file.path(orig_dir, "GBM_darmanis_rna_counts.feather"))
    write.csv(rna_df, file.path(orig_dir, "GBM_darmanis_rna_counts.csv"), row.names = FALSE)
    write.csv(celltype_df, file.path(orig_dir, "GBM_darmanis_celltype.csv"), row.names = FALSE)
    cat("Saved: raw counts and cell type\n")

    # Seurat standard normalization
    norm_mat <- as.matrix(rna_df[, -1])
    rownames(norm_mat) <- rna_df$pos

    seurat_tmp <- CreateSeuratObject(counts = norm_mat)
    seurat_tmp <- NormalizeData(seurat_tmp, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)

    norm_data <- as.matrix(GetAssayData(seurat_tmp, assay = "RNA", layer = "data"))

    # genes x cells
    norm_df <- as.data.frame(norm_data)
    norm_df <- rownames_to_column(norm_df, var = "pos")
    write_feather(norm_df, file.path(orig_dir, "GBM_darmanis_rna_seurat_norm.feather"))
    cat("Saved: GBM_darmanis_rna_seurat_norm.feather (genes x cells)\n")

    # cells x genes (transposed)
    norm_df_t <- as.data.frame(t(norm_data))
    norm_df_t <- rownames_to_column(norm_df_t, var = "pos")
    write_feather(norm_df_t, file.path(orig_dir, "GBM_darmanis_rna_seurat_norm_transposed.feather"))
    cat("Saved: GBM_darmanis_rna_seurat_norm_transposed.feather (cells x genes)\n")

    cat("Saved all files to:", orig_dir, "\n")
}

