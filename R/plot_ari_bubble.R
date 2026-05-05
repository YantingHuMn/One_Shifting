plot_ari_bubble <- function(read_dir, methods = c("kmeans", "leiden", "louvain")) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(ggplot2)
        library(dplyr)
        library(tidyr)
        library(cowplot)
    })

    all_data <- data.frame()
    for (method in methods) {
        path <- paste0(read_dir, "/", method, "/ARI_", method, "_best_norm.csv")
        df <- read.csv(path)
        df$criteria <- method
        all_data <- rbind(all_data, df)
    }

    all_data <- all_data %>% filter(!is.na(ARI))

    # calculate each criteria's rank
    all_data <- all_data %>%
        group_by(criteria) %>%
        mutate(rank = rank(-ARI, ties.method = "min")) %>%
        ungroup()

    # calculate mean rank and arrange
    method_stats <- all_data %>%
        group_by(method) %>%
        summarise(mean_rank = mean(rank)) %>%
        arrange(desc(mean_rank))

    method_order <- method_stats$method
    all_data$method <- factor(all_data$method, levels = method_order)
    method_stats$method <- factor(method_stats$method, levels = method_order)

    max_rank <- max(all_data$rank)

    # bar plot
    p_bar <- ggplot(method_stats, aes(x = mean_rank, y = method)) +
        geom_bar(stat = "identity", fill = "steelblue", width = 0.7) +
        scale_x_reverse() +
        labs(x = "Mean Rank", y = "") +
        theme_minimal() +
        theme(
            axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.grid.minor = element_blank(),
            plot.margin = margin(5.5, 0, 5.5, 5.5)
        )

    # bubble plot
    p_bubble <- ggplot(all_data, aes(x = criteria, y = method)) +
        geom_point(aes(size = max_rank - rank + 1, color = ARI)) +
        scale_size_continuous(
            name = "Rank\n(smaller = better)",
            range = c(2, 10),
            breaks = c(max_rank, max_rank * 0.75, max_rank * 0.5, max_rank * 0.25, 1),
            labels = c(1, round(max_rank * 0.25), round(max_rank * 0.5), round(max_rank * 0.75), max_rank)
        ) +
        scale_color_viridis_c(name = "ARI\n(larger = better)", option = "viridis") +
        labs(title = "Clustering Performance Comparison", x = "Clustering Criteria", y = "Method") +
        theme_minimal() +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1),
            axis.text.y = element_text(size = 8),
            panel.grid.major = element_line(color = "lightblue", linetype = "dashed"),
            legend.position = "right",
            plot.margin = margin(5.5, 5.5, 5.5, 0)
        )

    p_combined <- plot_grid(
        p_bar, p_bubble,
        nrow = 1,
        rel_widths = c(0.2, 0.8),
        align = "h",
        axis = "tb"
    )

    out_path <- paste0(read_dir, "/bubble_plot_ARI_comparison.png")
    ggsave(out_path, p_combined, width = 12, height = 10, dpi = 300)
    cat(paste0("[SAVE] ", out_path, "\n"))

    return(p_combined)
}