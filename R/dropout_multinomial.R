dropout_multinomial <- function(df, keep_par = 0.5, save_dir = NULL, cell_axis = c("row", "col")) {
    cell_axis <- match.arg(cell_axis)

    # separate pos column if present
    if (!is.numeric(df[[1]])) {
        pos_col <- df[[1]]
        mat <- as.matrix(df[, -1])
    } else {
        pos_col <- NULL
        mat <- as.matrix(df)
    }

    storage.mode(mat) <- "numeric"

    # dropout per cell
    if (cell_axis == "row") {
        # cells are rows
        for (i in seq_len(nrow(mat))) {
            lib_size <- sum(mat[i, ], na.rm = TRUE)
            if (lib_size == 0) next

            probs <- mat[i, ] / lib_size
            new_total <- round(lib_size * keep_par)
            mat[i, ] <- rmultinom(1, size = new_total, prob = probs)
        }
    } else {
        # cells are columns
        for (j in seq_len(ncol(mat))) {
            lib_size <- sum(mat[, j], na.rm = TRUE)
            if (lib_size == 0) next

            probs <- mat[, j] / lib_size
            new_total <- round(lib_size * keep_par)
            mat[, j] <- rmultinom(1, size = new_total, prob = probs)
        }
    }

    # calculate sparsity
    sparsity <- round(mean(mat == 0) * 100, 1)

    # rebuild df
    result <- as.data.frame(mat, check.names = FALSE)
    colnames(result) <- colnames(df)[seq_len(ncol(result)) + ifelse(is.null(pos_col), 0, 1)]

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