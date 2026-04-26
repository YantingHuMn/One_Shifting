run_summary_scatter_plot <- function(csv_path, method_name, skip_trans = NULL, skip_norm = NULL) {
    # Load libraries
    suppressPackageStartupMessages({
        library(ggplot2)
        library(dplyr)
    })
    output_path <- sub("\\.csv$", ".png", csv_path, ignore.case = TRUE)

    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    cat("Data loaded")

    if (!is.null(skip_trans)) {
        df <- df %>% filter(REP1_trans != skip_trans)
        cat(sprintf("Skipped trans: %s\n", skip_trans))
    }
    if (!is.null(skip_norm)) {
        df <- df %>% filter(REP1_norm != skip_norm)
        cat(sprintf("Skipped norm: %s\n", skip_norm))
    }

    has_xmean <- "x_mean" %in% colnames(df) && "residual_mean" %in% colnames(df)
    has_above <- "above" %in% colnames(df) && "below" %in% colnames(df)

    if (has_xmean) {
        x_col <- "x_mean"
        y_col <- "residual_mean"
        df$residual_mean <- -df$residual_mean
        x_label <- "x_mean (OUTPUT correlation mean)"
        y_label <- "residual_mean (OUTPUT - INPUT)"
    } else if (has_above) {
        x_col <- "below"
        y_col <- "above"
        x_label <- "Below (INPUT < OUTPUT)"
        y_label <- "Above (INPUT > OUTPUT)"
    } else {
        stop("CSV must have either (x_mean, residual_mean) or (above, below) columns")
    }

    cat(sprintf("\nUsing columns: %s vs %s\n", x_col, y_col))
    cat(sprintf("Unique trans: %s\n", paste(unique(df$REP1_trans), collapse = ", ")))
    cat(sprintf("Unique norm:  %s\n", paste(unique(df$REP1_norm), collapse = ", ")))

    all_colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#F781BF")
    all_shapes <- c(16, 17, 15, 18, 25, 21)

    unique_trans <- unique(df$REP1_trans)
    unique_norm  <- unique(df$REP1_norm)

    color_map <- setNames(all_colors[1:length(unique_trans)], unique_trans)
    shape_map <- setNames(all_shapes[1:length(unique_norm)],  unique_norm)

    df$REP1_trans <- factor(df$REP1_trans, levels = unique_trans)
    df$REP1_norm  <- factor(df$REP1_norm,  levels = unique_norm)

    n_points <- nrow(df)

    p <- ggplot(df, aes(x = .data[[x_col]], y = .data[[y_col]], color = REP1_trans, shape = REP1_norm)) +
        geom_point(size = 3.5, stroke = 0.8, alpha = 0.5) +
        scale_color_manual(values = color_map) +
        scale_shape_manual(values = shape_map) +
        labs(
            x = x_label,
            y = y_label,
            title = paste0(method_name, ": ", basename(csv_path)),
            subtitle = sprintf("n = %d", n_points),
            color = "Trans",
            shape = "Norm"
        ) +
        theme_bw() +
        theme(
            legend.position = "right",
            plot.title = element_text(size = 11, hjust = 0.5),
            legend.text = element_text(size = 9),
            legend.title = element_text(size = 10, face = "bold")
        )

    if (has_xmean) {
        p <- p +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray", linewidth = 0.5) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "gray", linewidth = 0.5)
    } else {
        p <- p +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray", linewidth = 0.5)
    }

    ggsave(output_path, p, width = 9, height = 6, dpi = 300)
    cat(sprintf("Saved to %s\n", output_path))
}

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
method_name <- args[2]
skip_trans <- if (length(args) >= 3 && args[3] != "") args[3] else NULL
skip_norm <- if (length(args) >= 4 && args[4] != "") args[4] else NULL


run_summary_scatter_plot(csv_path, method_name, skip_trans, skip_norm)