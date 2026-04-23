# Split expression data into train/val/test sets for TF and HVG targets
split_tf_hvg_data <- function(expr_df_path, output_dir, tf_names, hvg_names, test_frac = 0.2, val_frac = 0.1, seed = 42) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(Seurat)
        library(arrow)
    })

    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    # If tf_names/hvg_names are file paths, read them
    if (length(tf_names) == 1 && file.exists(tf_names)) {
        tf_names <- readLines(tf_names)
    }
    if (length(hvg_names) == 1 && file.exists(hvg_names)) {
        hvg_names <- readLines(hvg_names)
    }

    expr_df <- read_feather(expr_df_path)

    # Remove non-gene columns
    drop_cols <- intersect(colnames(expr_df), c("pos", "index", "Unnamed: 0"))
    if (length(drop_cols) > 0) {
        expr_df <- expr_df[, !(colnames(expr_df) %in% drop_cols), drop = FALSE]
    }

    # Already cells x genes, no transpose needed
    expr_mat <- as.data.frame(expr_df)

    n <- nrow(expr_mat)
    cat(sprintf("[INFO] Expression matrix: %d cells x %d genes\n", n, ncol(expr_mat)))

    tf_names <- intersect(tf_names, colnames(expr_mat))
    hvg_names <- intersect(hvg_names, colnames(expr_mat))
    cat(sprintf("[INFO] TFs matched: %d, HVGs matched: %d\n",
                length(tf_names), length(hvg_names)))

    tf_matrix <- expr_mat[, tf_names, drop = FALSE]
    target_matrix <- expr_mat[, hvg_names, drop = FALSE]

    # Train/Val/Test split
    set.seed(seed)
    n_test <- round(n * test_frac)
    n_val <- round(n * val_frac)

    all_idx <- seq_len(n)
    test_idx <- sort(sample(all_idx, n_test))
    remain <- setdiff(all_idx, test_idx)
    val_idx <- sort(sample(remain, n_val))
    train_idx <- setdiff(remain, val_idx)

    cat(sprintf("[INFO] Split: %d train, %d val, %d test\n", length(train_idx), length(val_idx), length(test_idx)))

    write_feather(tf_matrix[train_idx, , drop = FALSE], file.path(output_dir, "tf_train.feather"))
    write_feather(tf_matrix[val_idx, , drop = FALSE], file.path(output_dir, "tf_val.feather"))
    write_feather(tf_matrix[test_idx, , drop = FALSE], file.path(output_dir, "tf_test.feather"))
    write_feather(target_matrix[train_idx, , drop = FALSE], file.path(output_dir, "target_train.feather"))
    write_feather(target_matrix[val_idx, , drop = FALSE], file.path(output_dir, "target_val.feather"))
    write_feather(target_matrix[test_idx, , drop = FALSE], file.path(output_dir, "target_test.feather"))

    write.csv(data.frame(idx = train_idx), file.path(output_dir, "train_indices.csv"), row.names = FALSE)
    write.csv(data.frame(idx = val_idx), file.path(output_dir, "val_indices.csv"),   row.names = FALSE)
    write.csv(data.frame(idx = test_idx), file.path(output_dir, "test_indices.csv"),  row.names = FALSE)

    cat(sprintf("\n[DONE] Saved to %s\n", output_dir))

    return(invisible(list(
        tf_train = tf_matrix[train_idx, , drop = FALSE],
        tf_val = tf_matrix[val_idx, , drop = FALSE],
        tf_test = tf_matrix[test_idx, , drop = FALSE],
        target_train = target_matrix[train_idx, , drop = FALSE],
        target_val = target_matrix[val_idx, , drop = FALSE],
        target_test = target_matrix[test_idx, , drop = FALSE],
        train_idx = train_idx,
        val_idx = val_idx,
        test_idx = test_idx
    )))
}