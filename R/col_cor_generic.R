col_cor_generic <- function(X, Y, method = "pearson") {
    stopifnot(all(dim(X) == dim(Y)))
    p <- ncol(X)
    r_values <- numeric(p)

    for (j in 1:p) {
        x_col <- X[, j]
        y_col <- Y[, j]

        valid_idx <- !is.na(x_col) & !is.na(y_col)
        x_valid <- x_col[valid_idx]
        y_valid <- y_col[valid_idx]

        if (length(x_valid) > 1) {
            if (sd(x_valid) > 0 && sd(y_valid) > 0) {
                r_values[j] <- cor(x_valid, y_valid, method = method)
            } else {
                r_values[j] <- NA_real_
            }
        } else {
            r_values[j] <- NA_real_
        }
    }

    names(r_values) <- colnames(X)
    return(r_values)
}