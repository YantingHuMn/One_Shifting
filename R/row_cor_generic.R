# Helper: Row-wise Pearson correlation
row_cor_generic <- function(X, Y, method = "pearson") {
    stopifnot(all(dim(X) == dim(Y)))
    n <- nrow(X)
    requested_cores <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")))
    if (is.na(requested_cores) || requested_cores < 1) requested_cores <- 1L

    cor_one_row <- function(i) {
        x_row <- X[i, ]
        y_row <- Y[i, ]

        valid_idx <- !is.na(x_row) & !is.na(y_row)
        x_valid <- x_row[valid_idx]
        y_valid <- y_row[valid_idx]

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
        parallel::mclapply(seq_len(n), cor_one_row,
                           mc.cores = min(requested_cores, max(1L, n)),
                           mc.preschedule = TRUE),
        use.names = FALSE
    )

    # If no rownames exist, create default names
    if (is.null(rownames(X))) {
        names(r_values) <- paste0("row_", 1:n)
    } else {
        names(r_values) <- rownames(X)
    }

    return(r_values)
}
