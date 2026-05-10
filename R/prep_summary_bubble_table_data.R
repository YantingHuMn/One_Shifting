prep_summary_bubble_table_data <- function(method, data_type, base_dir, with_col, norm_info_path, out_csv, metric, v2_trans_factor, v2_norm_factor, trans_factors) {
    best_norm <- read.csv(norm_info_path)
    cat(sprintf("[INFO] method=%s, data_type=%s, metric=%s\n", method, data_type, metric))

    results <- data.frame()

    for (trans_val in trans_factors) {
        row <- best_norm[best_norm$REP1_trans == trans_val, ]
        if (nrow(row) == 0) {
            cat(sprintf("  [SKIP] No norm found for trans=%s\n", trans_val))
            next
        }
        row <- row[1, ]

        results <- rbind(results, data.frame(
            method = method,
            data_type = data_type,
            metric = metric,
            trans = trans_val,
            norm = as.character(row$REP1_norm),
            mean_performance = round(row$x_mean, 6),
            mean_residual = round(row$residual_mean, 6),
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