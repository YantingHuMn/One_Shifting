plot_summary_bubble_table <- function(csv_path, filter_method = NULL) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(ggplot2)
        library(dplyr)
        library(tidyr)
        library(RColorBrewer)
        library(cowplot)
    })

    base_path <- sub("\\.csv$", "", csv_path)
    df <- read.csv(csv_path)

    if (!is.null(filter_method)) {
        df <- df %>% filter(tolower(method) != tolower(filter_method))
        if (nrow(df) == 0) {
            cat("No data left after excluding method:", filter_method, "\n")
            return(invisible(NULL))
        }
        base_path <- paste0(base_path, "_no_", filter_method)
        cat("Excluded method:", filter_method, "->", nrow(df), "rows remaining\n")
    }

    unique_data_types <- unique(df$data_type)
    title_suffix <- if (length(unique_data_types) == 1) paste0(' for "', unique_data_types[1], '"') else ""

    x_combinations <- df %>%
        select(method, data_type, metric) %>%
        distinct() %>%
        mutate(x_id = row_number())

    df <- df %>%
        left_join(x_combinations, by = c("method", "data_type", "metric"))

    df <- df %>%
        group_by(x_id) %>%
        mutate(
            rank_perf = rank(-mean_performance, ties.method = "average"),
            rank_residual = rank(mean_residual, ties.method = "average"),
            combined_rank = rank_perf + rank_residual
        ) %>%
        ungroup()

    avg_rank_perf <- df %>%
        group_by(trans) %>%
        summarise(avg_rank = mean(rank_perf, na.rm = TRUE))

    avg_rank_resid <- df %>%
        group_by(trans) %>%
        summarise(avg_rank = mean(rank_residual, na.rm = TRUE))

    # Order: best (smallest avg_rank) on top
    perf_order <- avg_rank_perf %>% arrange(avg_rank) %>% pull(trans) %>% as.character()
    resid_order <- avg_rank_resid %>% arrange(avg_rank) %>% pull(trans) %>% as.character()

    df_perf <- df %>% mutate(trans = factor(trans, levels = rev(perf_order)))
    df_resid <- df %>% mutate(trans = factor(trans, levels = rev(resid_order)))
    avg_rank_perf$trans <- factor(avg_rank_perf$trans, levels = rev(perf_order))
    avg_rank_resid$trans <- factor(avg_rank_resid$trans, levels = rev(resid_order))

    avg_rank_perf <- avg_rank_perf %>%
        mutate(bar_len = max(avg_rank) - avg_rank + min(avg_rank))
    avg_rank_resid <- avg_rank_resid %>%
        mutate(bar_len = max(avg_rank) - avg_rank + min(avg_rank))

    # Order x-axis by best trans value for each plot
    best_perf_trans <- perf_order[1]
    perf_x_order <- df_perf %>%
        filter(trans == best_perf_trans) %>%
        arrange(desc(mean_performance)) %>%
        pull(x_id) %>%
        unique()
    perf_x_order <- c(perf_x_order, setdiff(unique(df_perf$x_id), perf_x_order))

    best_resid_trans <- resid_order[1]
    resid_x_order <- df_resid %>%
        filter(trans == best_resid_trans) %>%
        arrange(mean_residual) %>%
        pull(x_id) %>%
        unique()
    resid_x_order <- c(resid_x_order, setdiff(unique(df_resid$x_id), resid_x_order))

    x_labels_df <- x_combinations %>%
        mutate(label = paste(method, data_type, metric, sep = " | "))

    perf_x_labels <- x_labels_df$label[match(perf_x_order, x_labels_df$x_id)]
    resid_x_labels <- x_labels_df$label[match(resid_x_order, x_labels_df$x_id)]

    df_perf$x_id <- factor(df_perf$x_id, levels = perf_x_order)
    df_resid$x_id <- factor(df_resid$x_id, levels = resid_x_order)

    # --- Plot 1: mean_performance ---
    p1_main <- ggplot(df_perf, aes(x = x_id, y = trans)) +
        geom_point(aes(size = rank_perf, color = mean_performance), alpha = 0.9) +
        scale_size_continuous(
            name = "Rank\n(smaller = better)",
            range = c(12, 1.5),
            guide = guide_legend(order = 2)
        ) +
        scale_color_viridis_c(
            name = "Mean Performance\n(larger = better)",
            option = "D", direction = 1,
            guide = guide_colorbar(order = 1)
        ) +
        scale_x_discrete(labels = perf_x_labels) +
        theme_minimal() +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            axis.text.y = element_text(size = 10),
            panel.grid.major = element_line(color = "grey90"),
            panel.grid.minor = element_blank(),
            legend.position = "right",
            plot.title = element_text(hjust = 0.5, face = "bold")
        ) +
        labs(x = "Method Combination", y = "Transformation",
             title = paste0("Mean Performance", title_suffix))

    p1_bar <- ggplot(avg_rank_perf, aes(x = bar_len, y = trans)) +
        geom_bar(stat = "identity", fill = "steelblue", width = 0.7) +
        scale_x_reverse() +
        theme_minimal() +
        theme(
            axis.text.y = element_blank(), axis.title.y = element_blank(),
            axis.ticks.y = element_blank(), axis.text.x = element_blank(),
            axis.title.x = element_blank(), axis.ticks.x = element_blank(),
            panel.grid = element_blank()
        )

    plot1 <- plot_grid(p1_bar, p1_main, nrow = 1, rel_widths = c(0.15, 0.85),
                       align = "h", axis = "tb")

    # --- Plot 2: mean_residual ---
    p2_main <- ggplot(df_resid, aes(x = x_id, y = trans)) +
        geom_point(aes(size = rank_residual, color = mean_residual), alpha = 0.9) +
        scale_size_continuous(
            name = "Rank\n(smaller = better)",
            range = c(12, 1.5),
            guide = guide_legend(order = 2)
        ) +
        scale_color_viridis_c(
            name = "Mean Residual\n(smaller = better)",
            option = "magma", direction = -1,
            guide = guide_colorbar(order = 1)
        ) +
        scale_x_discrete(labels = resid_x_labels) +
        theme_minimal() +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            axis.text.y = element_text(size = 10),
            panel.grid.major = element_line(color = "grey90"),
            panel.grid.minor = element_blank(),
            legend.position = "right",
            plot.title = element_text(hjust = 0.5, face = "bold")
        ) +
        labs(x = "Method Combination", y = "Transformation",
             title = paste0("Mean Residual", title_suffix))

    p2_bar <- ggplot(avg_rank_resid, aes(x = bar_len, y = trans)) +
        geom_bar(stat = "identity", fill = "coral", width = 0.7) +
        scale_x_reverse() +
        theme_minimal() +
        theme(
            axis.text.y = element_blank(), axis.title.y = element_blank(),
            axis.ticks.y = element_blank(), axis.text.x = element_blank(),
            axis.title.x = element_blank(), axis.ticks.x = element_blank(),
            panel.grid = element_blank()
        )

    plot2 <- plot_grid(p2_bar, p2_main, nrow = 1, rel_widths = c(0.15, 0.85),
                       align = "h", axis = "tb")

    ggsave(paste0(base_path, "_performance.png"), plot1,
           width = 14, height = 10, dpi = 300)
    ggsave(paste0(base_path, "_residual.png"), plot2,
           width = 14, height = 10, dpi = 300)
    write.csv(df, paste0(base_path, "_table_df.csv"), row.names = FALSE)
    write.csv(avg_rank_perf, paste0(base_path, "_avg_rank_perf.csv"), row.names = FALSE)
    write.csv(avg_rank_resid, paste0(base_path, "_avg_rank_resid.csv"), row.names = FALSE)

    cat("Plots saved!\n")
}