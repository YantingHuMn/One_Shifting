cut_df_pct <- function(mat_num, mat2_num = NULL, zero_pct_max = NULL, pearson_min = NULL, spearman_min = NULL, lib_size_min = NULL) {
    keep <- rep(TRUE, ncol(mat_num))

    # Helper: if value is "pXX", compute that percentile from vec; otherwise use as numeric
    resolve_threshold <- function(val, vec, lower_tail = TRUE) {
        if (is.character(val) && grepl("^p[0-9]+(\\.[0-9]+)?$", val, ignore.case = TRUE)) {
            pct <- as.numeric(sub("^p", "", val, ignore.case = TRUE))
            thresh <- quantile(vec, probs = pct / 100, na.rm = TRUE, names = FALSE)
            cat(sprintf("  Resolved '%s' to %.4f (from distribution)\n", val, thresh))
            return(thresh)
        }
        return(as.numeric(val))
    }

    if (!is.null(zero_pct_max)) {
        zero_pct <- colMeans(mat_num == 0) * 100
        threshold <- resolve_threshold(zero_pct_max, zero_pct)
        keep <- keep & (zero_pct <= threshold)
        cat(sprintf("Zero pct filter (<= %.2f%%): %d / %d columns kept\n",
                    threshold, sum(keep), length(keep)))
    }

    if (!is.null(lib_size_min)) {
        lib_sizes <- colSums(mat_num)
        threshold <- resolve_threshold(lib_size_min, lib_sizes)
        keep <- keep & (lib_sizes >= threshold)
        cat(sprintf("Library size filter (>= %.2f): %d / %d columns kept\n",
                    threshold, sum(keep), length(keep)))
    }

    if (!is.null(pearson_min)) {
        if (is.null(mat2_num)) stop("pearson_min requires mat2_num")
        cor_vec <- sapply(seq_len(ncol(mat_num)), function(i) cor(mat_num[, i], mat2_num[, i], method = "pearson"))
        threshold <- resolve_threshold(pearson_min, cor_vec)
        keep <- keep & (!is.na(cor_vec) & cor_vec >= threshold)
        cat(sprintf("Pearson filter (>= %.4f): %d / %d columns kept\n",
                    threshold, sum(keep), length(keep)))
    }

    if (!is.null(spearman_min)) {
        if (is.null(mat2_num)) stop("spearman_min requires mat2_num")
        cor_vec <- sapply(seq_len(ncol(mat_num)), function(i) cor(mat_num[, i], mat2_num[, i], method = "spearman"))
        threshold <- resolve_threshold(spearman_min, cor_vec)
        keep <- keep & (!is.na(cor_vec) & cor_vec >= threshold)
        cat(sprintf("Spearman filter (>= %.4f): %d / %d columns kept\n",
                    threshold, sum(keep), length(keep)))
    }

    cat(sprintf("Cut columns: %d kept out of %d\n", sum(keep), length(keep)))
    return(keep)
}
