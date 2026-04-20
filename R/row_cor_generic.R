# Helper: Row-wise Pearson correlation
row_cor_generic <- function(X, Y, method = "pearson") {
    stopifnot(all(dim(X) == dim(Y)))
    n <- nrow(X)
    r_values <- numeric(n)

    for (i in 1:n) {
        x_row <- X[i, ]
        y_row <- Y[i, ]

        valid_idx <- !is.na(x_row) & !is.na(y_row)
        x_valid <- x_row[valid_idx]
        y_valid <- y_row[valid_idx]

        if (length(x_valid) > 1) {
            if (sd(x_valid) > 0 && sd(y_valid) > 0) {
                r_values[i] <- cor(x_valid, y_valid, method = method)
            } else {
                r_values[i] <- NA_real_
            }
        } else {
            r_values[i] <- NA_real_
        }
    }

    # If no rownames exist, create default names
    if (is.null(rownames(X))) {
        names(r_values) <- paste0("row_", 1:n)
    } else {
        names(r_values) <- rownames(X)
    }

    return(r_values)
}
