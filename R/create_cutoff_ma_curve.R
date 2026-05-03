create_cutoff_ma_curve <- function(plot_data, cutoff_seq, trans_val, norm_val, metric_type, V1, y_label, x_label) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(ggplot2)
    })

    n_total <- nrow(plot_data)
    na_df1 <- sum(is.na(plot_data$df1_VAE))
    na_df2 <- sum(is.na(plot_data$df2_VAE))
    na_both <- sum(is.na(plot_data$df1_VAE) & is.na(plot_data$df2_VAE))
    na_any <- sum(is.na(plot_data$df1_VAE) | is.na(plot_data$df2_VAE))

    if (na_any > 0) {
        cat("  WARNING: NA detected in plot_data (trans=", trans_val, "):\n")
        cat("    Total rows:", n_total, "\n")
        cat("    NA in df1_VAE (x):", na_df1, "\n")
        cat("    NA in df2_VAE (y):", na_df2, "\n")
        cat("    NA in both:", na_both, "\n")
        cat("    Rows dropped:", na_any, "\n")
        cat("    Rows remaining:", n_total - na_any, "\n")
    }

    plot_data <- plot_data[!is.na(plot_data$df1_VAE) & !is.na(plot_data$df2_VAE), ]

    results <- data.frame(cutoff = cutoff_seq, pct_below = NA, n_points = NA)

    for (i in seq_along(cutoff_seq)) {
        c <- cutoff_seq[i]
        filtered <- plot_data[plot_data$df1_VAE > c | plot_data$df2_VAE > c, ]
        n <- nrow(filtered)
        if (n > 0) {
            results$pct_below[i] <- sum(filtered$df2_VAE < filtered$df1_VAE) / n * 100
            results$n_points[i] <- n
        }
    }

    results <- results[!is.na(results$pct_below), ]

    p <- ggplot(results, aes(x = cutoff, y = pct_below)) +
        geom_line(linewidth = 0.8) +
        geom_hline(yintercept = 50, linetype = "dashed", color = "red") +
        labs(x = paste0(metric_type, " Cutoff"), y = paste("% below diagonal: ", x_label, "/", y_label),
             title = paste0(metric_type, ": ", V1, " ", trans_val, "; ", norm_val)) +
        scale_x_continuous(limits = c(-0.5, 1)) +
        scale_y_continuous(limits = c(0, 100)) +
        theme_bw() +
        theme(
            plot.subtitle = element_text(size = 9, color = "gray30")
        )

    return(p)
}