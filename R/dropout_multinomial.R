dropout_multinomial <- function(df, keep_par = 0.5, save_dir = NULL) {
    # separate pos column if present
    if (!is.numeric(df[[1]])) {
        pos_col <- df[[1]]
        mat <- as.matrix(df[, -1])
    } else {
        pos_col <- NULL
        mat <- as.matrix(df)
    }

    # dropout per cell (row)
    for (i in seq_len(nrow(mat))) {
        lib_size <- sum(mat[i, ])
        if (lib_size == 0) next

        probs <- mat[i, ] / lib_size
        new_total <- round(lib_size * keep_par)
        mat[i, ] <- rmultinom(1, size = new_total, prob = probs)
    }

    # calculate sparsity
    sparsity <- round(mean(mat == 0) * 100, 1)

    # rebuild df
    result <- as.data.frame(mat)
    if (!is.null(pos_col)) {
        result <- cbind(pos = pos_col, result)
    }

    if (!is.null(save_dir)) {
        library(arrow)
        dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
        write_feather(result, file.path(save_dir, paste0("dropout_", keep_par, "_sparsity_", sparsity, ".feather")))
    }

    return(result)
}