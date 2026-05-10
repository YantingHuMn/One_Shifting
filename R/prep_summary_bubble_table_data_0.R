prep_summary_bubble_table_data <- function(method, data_type, base_dir, with_col, norm_info_path, out_csv, metric, v2_trans_factor, v2_norm_factor, trans_factors) {
    best_norm <- read.csv(norm_info_path)
    cat(sprintf("[INFO] method=%s, data_type=%s, with_col=%s, metric=%s\n", method, data_type, with_col, metric))
    cat(sprintf("[INFO] base_dir=%s\n", base_dir))

    prefix <- if (metric == "pearson") "d_pearson" else "g_spearman"

    # Detect noGTzero from norm_info_path or out_csv
    noGTzero <- grepl("_noGTzero", norm_info_path) || grepl("_noGTzero", out_csv)
    gt_tag <- if (noGTzero) "_noGTzero" else ""

    results <- data.frame()

    for (trans_val in trans_factors) {

        v1_norm <- best_norm$REP1_norm[best_norm$REP1_trans == trans_val]
        if (length(v1_norm) == 0) {
            cat(sprintf("  [SKIP] No norm found for trans=%s\n", trans_val))
            next
        }
        v1_norm <- as.character(v1_norm[1])

        if (with_col) {
            fig_dir <- list.dirs(base_dir, recursive = FALSE, full.names = FALSE)
            fig_dir <- fig_dir[grepl("^Figures_col", fig_dir)][1]
        } else {
            fig_dir <- list.dirs(base_dir, recursive = FALSE, full.names = FALSE)
            fig_dir <- fig_dir[grepl("^Figures_row", fig_dir)][1]
        }

        path1 <- file.path(base_dir, fig_dir, paste0(
            prefix, gt_tag, "_", ifelse(with_col, "col", "row"), "_scatter_v1_trans_", trans_val,
            "_norm_", v1_norm,
            "_v2_trans_", v2_trans_factor,
            "_norm_", v2_norm_factor, ".csv"))

        if (!file.exists(path1)) {
            cat(sprintf("  [SKIP] %s not found: %s\n", metric, path1))
            next
        }

        df1 <- read.csv(path1)

        mean_performance <- mean(df1$VAE, na.rm = TRUE)
        mean_residual <- mean(df1$residual, na.rm = TRUE)
        n_genes <- nrow(df1)

        results <- rbind(results, data.frame(
            method = method,
            data_type = data_type,
            metric = metric,
            trans = trans_val,
            norm = v1_norm,
            mean_performance = round(mean_performance, 6),
            mean_residual = round(mean_residual, 6),
            n_genes = n_genes,
            stringsAsFactors = FALSE
        ))
    }

    if (nrow(results) > 0) {
        write_header <- !file.exists(out_csv)
        write.table(results, out_csv, append = TRUE, sep = ",",
                    row.names = FALSE, col.names = write_header, quote = FALSE)
        cat(sprintf("[SAVE] %s written to: %s\n",
                    ifelse(write_header, "Created", "Appended"), out_csv))
    } else {
        cat("[WARNING] No results to write\n")
    }
}