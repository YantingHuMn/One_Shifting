select_tf_hvg <- function(expr_df_path, output_dir = NULL, tf_list_path = "../One_Shifting/data/DatabaseExtract_v_1.01.csv", n_hvg = 1000) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(Seurat)
        library(arrow)
    })

    if (!is.null(output_dir)) {
        dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    }

    # tf_list_path download from https://humantfs.ccbr.utoronto.ca/download.php

    # 1. Parse expression matrix: cell by gene
    expr_df <- read_feather(expr_df_path)

    # Remove non-gene columns
    drop_cols <- intersect(colnames(expr_df), c("pos", "index", "Unnamed: 0"))
    if (length(drop_cols) > 0) {
        expr_df <- expr_df[, !(colnames(expr_df) %in% drop_cols), drop = FALSE]
    }

    gene_names <- colnames(expr_df)
    # Seurat wants genes x cells
    expr_matrix <- t(as.matrix(expr_df))

    cat(sprintf("[INFO] Expression matrix: %d genes x %d cells\n", nrow(expr_matrix), ncol(expr_matrix)))

    # 2. Read TF list 
    cat(sprintf("[INFO] Reading TF list: %s\n", tf_list_path))
    tf_db <- read.csv(tf_list_path, stringsAsFactors = FALSE)

    if ("HGNC.symbol" %in% colnames(tf_db)) {
        if ("Is.TF." %in% colnames(tf_db)) {
            tf_names <- tf_db$HGNC.symbol[tf_db$Is.TF. == "Yes"]
        } else {
            tf_names <- tf_db$HGNC.symbol
        }
    } else if ("Name" %in% colnames(tf_db)) {
        tf_names <- tf_db$Name
    } else if ("Gene" %in% colnames(tf_db)) {
        tf_names <- tf_db$Gene
    } else {
        tf_names <- tf_db[[1]]
    }
    tf_names <- unique(tf_names[tf_names != ""])

    tf_in_data <- intersect(gene_names, tf_names)
    cat(sprintf("[INFO] TF list: %d total, %d matched in data\n",
                length(tf_names), length(tf_in_data)))

    # 3. Select HVGs via Seurat
    cat(sprintf("[INFO] Selecting %d HVGs with Seurat VST...\n", n_hvg))

    sobj <- CreateSeuratObject(counts = expr_matrix)
    sobj <- NormalizeData(sobj, verbose = FALSE)
    sobj <- FindVariableFeatures(sobj, selection.method = "vst", nfeatures = n_hvg, verbose = FALSE)

    hvg_all <- VariableFeatures(sobj)

    # Remove TFs from HVG list
    hvg_non_tf <- setdiff(hvg_all, tf_in_data)
    n_tf_in_hvg <- length(intersect(hvg_all, tf_in_data))

    cat(sprintf("[INFO] HVGs found: %d (%d were TFs, removed)\n",
                length(hvg_all), n_tf_in_hvg))
    cat(sprintf("[INFO] Final: %d TFs (input), %d HVGs (target)\n",
                length(tf_in_data), length(hvg_non_tf)))

    if (!is.null(output_dir)) {
        writeLines(tf_in_data,  file.path(output_dir, "tf_names.txt"))
        writeLines(hvg_non_tf,  file.path(output_dir, "hvg_target_names.txt"))
    }

    return(list(
        tf_names = tf_in_data,
        hvg_target_names = hvg_non_tf
    ))
}
