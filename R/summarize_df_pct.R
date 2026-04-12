summarize_df_pct <- function(feather_path, feather_path2 = NULL, want = c("zero_percentage"), quantiles = NULL) {
    suppressPackageStartupMessages({
        library(arrow)
    })

    mat <- read_feather(feather_path)
    if (!is.numeric(mat[[1]])) mat <- mat[, -1]
    mat <- as.matrix(mat)

    if (!is.null(feather_path2)) {
        mat2 <- read_feather(feather_path2)
        if (!is.numeric(mat2[[1]])) mat2 <- mat2[, -1]
        mat2 <- as.matrix(mat2)
        stopifnot(nrow(mat) == nrow(mat2), ncol(mat) == ncol(mat2))
    }

    # --- zero_percentage ---
    if ("zero_percentage" %in% want) {
        zero_pct <- colMeans(mat == 0) * 100
        cat("=== Zero Percentage Summary (per column) ===\n")
        print(summary(zero_pct))
        if (!is.null(quantiles)) {
            q_vals <- quantile(zero_pct, probs = quantiles)
            cat("\n=== Zero Percentage at Given Quantiles ===\n")
            for (i in seq_along(quantiles)) {
                cat(sprintf("  %s percentile: %.2f%% zeros\n",
                            names(q_vals)[i], q_vals[i]))
            }
        }
        cat("\n")
    }

    # --- library_size ---
    if ("library_size" %in% want) {
        lib_sizes <- colSums(mat)
        cat("=== Library Size Summary (per column) ===\n")
        print(summary(lib_sizes))
        if (!is.null(quantiles)) {
            q_vals <- quantile(lib_sizes, probs = quantiles)
            cat("\n=== Library Size at Given Quantiles ===\n")
            for (i in seq_along(quantiles)) {
                cat(sprintf("  %s percentile: %.2f\n",
                            names(q_vals)[i], q_vals[i]))
            }
        }
        cat("\n")
    }

    # --- pearson_correlation ---
    if ("pearson_correlation" %in% want) {
        if (is.null(feather_path2)) stop("pearson_correlation requires feather_path2")
        cor_vec <- sapply(seq_len(ncol(mat)), function(i) cor(mat[, i], mat2[, i], method = "pearson"))
        cat("=== Pearson Correlation Summary (per column) ===\n")
        print(summary(cor_vec))
        if (!is.null(quantiles)) {
            q_vals <- quantile(cor_vec, probs = quantiles, na.rm = TRUE)
            cat("\n=== Pearson Correlation at Given Quantiles ===\n")
            for (i in seq_along(quantiles)) {
                cat(sprintf("  %s percentile: %.4f\n",
                            names(q_vals)[i], q_vals[i]))
            }
        }
        cat("\n")
    }

    # --- spearman_correlation ---
    if ("spearman_correlation" %in% want) {
        if (is.null(feather_path2)) stop("spearman_correlation requires feather_path2")
        cor_vec <- sapply(seq_len(ncol(mat)), function(i) cor(mat[, i], mat2[, i], method = "spearman"))
        cat("=== Spearman Correlation Summary (per column) ===\n")
        print(summary(cor_vec))
        if (!is.null(quantiles)) {
            q_vals <- quantile(cor_vec, probs = quantiles, na.rm = TRUE)
            cat("\n=== Spearman Correlation at Given Quantiles ===\n")
            for (i in seq_along(quantiles)) {
                cat(sprintf("  %s percentile: %.4f\n",
                            names(q_vals)[i], q_vals[i]))
            }
        }
        cat("\n")
    }
}