col_cor_generic <- function(X, Y, method = "pearson") {
    stopifnot(all(dim(X) == dim(Y)))
    p <- ncol(X)
    requested_cores <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")))
    if (is.na(requested_cores) || requested_cores < 1) requested_cores <- 1L

    cor_one_col <- function(j) {
        x_col <- X[, j]
        y_col <- Y[, j]

        valid_idx <- !is.na(x_col) & !is.na(y_col)
        x_valid <- x_col[valid_idx]
        y_valid <- y_col[valid_idx]

        if (length(x_valid) > 1) {
            if (sd(x_valid) > 0 && sd(y_valid) > 0) {
                return(cor(x_valid, y_valid, method = method))
            } else {
                return(NA_real_)
            }
        } else {
            return(NA_real_)
        }
    }

    r_values <- unlist(
        parallel::mclapply(seq_len(p), cor_one_col,
                           mc.cores = min(requested_cores, max(1L, p)),
                           mc.preschedule = TRUE),
        use.names = FALSE
    )

    names(r_values) <- colnames(X)
    return(r_values)
}