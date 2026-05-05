plot_ari_bubble_trans <- function(read_dir, methods = c("kmeans", "leiden", "louvain")) {
    suppressPackageStartupMessages({
        library(ggplot2)
        library(dplyr)
        library(tidyr)
        library(cowplot)
    })

    target_prefixes <- c("VAE", "DCA_mse", "scVI_mse", "Transformer_denoise")
    norm_patterns <- c("_no_norm$", "_standardize$", "_1000000$", "_100000$", "_10000$", "_1000$")

    parse_method_name <- function(m) {
        for (pfx in target_prefixes) {
            if (startsWith(m, paste0(pfx, "_"))) {
                suffix <- sub(paste0("^", pfx, "_"), "", m)
                trans <- suffix
                for (np in norm_patterns) {
                    if (grepl(np, suffix)) {
                        trans <- sub(np, "", suffix)
                        break
                    }
                }
                return(data.frame(prefix = pfx, trans = trans, stringsAsFactors = FALSE))
            }
        }
        return(NULL)
    }

    all_data <- data.frame()
    for (cm in methods) {
        path <- file.path(read_dir, cm, paste0("ARI_", cm, "_best_norm.csv"))
        df <- read.csv(path)
        df$criteria <- cm

        parsed <- do.call(rbind, lapply(df$method, function(m) parse_method_name(m)))
        if (is.null(parsed) || nrow(parsed) == 0) next

        keep <- !sapply(seq_len(nrow(df)), function(i) is.null(parse_method_name(df$method[i])))
        df <- df[keep, ]
        parsed <- do.call(rbind, lapply(df$method, function(m) parse_method_name(m)))

        df$prefix <- parsed$prefix
        df$trans <- parsed$trans
        all_data <- rbind(all_data, df)
    }

    all_data <- all_data %>% filter(!is.na(ARI))

    make_bubble <- function(plot_df, plot_title, out_path) {
        plot_df <- plot_df %>%
            group_by(prefix) %>%
            mutate(rank_ari = rank(-ARI, ties.method = "average")) %>%
            ungroup()

        avg_rank <- plot_df %>%
            group_by(trans) %>%
            summarise(avg_rank = mean(rank_ari, na.rm = TRUE)) %>%
            arrange(avg_rank)

        trans_order <- avg_rank %>% pull(trans) %>% as.character()
        plot_df$trans <- factor(plot_df$trans, levels = rev(trans_order))
        avg_rank$trans <- factor(avg_rank$trans, levels = rev(trans_order))
        avg_rank <- avg_rank %>%
            mutate(bar_len = max(avg_rank) - avg_rank + min(avg_rank))

        p_bar <- ggplot(avg_rank, aes(x = bar_len, y = trans)) +
            geom_bar(stat = "identity", fill = "steelblue", width = 0.7) +
            scale_x_reverse() +
            theme_minimal() +
            theme(
                axis.text.y = element_blank(), axis.title.y = element_blank(),
                axis.ticks.y = element_blank(), axis.text.x = element_blank(),
                axis.title.x = element_blank(), axis.ticks.x = element_blank(),
                panel.grid = element_blank()
            )

        p_main <- ggplot(plot_df, aes(x = prefix, y = trans)) +
            geom_point(aes(size = rank_ari, color = ARI), alpha = 0.9) +
            scale_size_continuous(
                name = "Rank\n(smaller = better)",
                range = c(8, 3),
                guide = guide_legend(order = 2)
            ) +
            scale_color_viridis_c(
                name = "ARI\n(larger = better)",
                option = "D", direction = 1,
                guide = guide_colorbar(order = 1)
            ) +
            theme_minimal() +
            theme(
                axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
                axis.text.y = element_text(size = 10),
                panel.grid.major = element_line(color = "grey90"),
                panel.grid.minor = element_blank(),
                legend.position = "right",
                plot.title = element_text(hjust = 0.5, face = "bold")
            ) +
            labs(x = "Method", y = "Transformation", title = plot_title)

        p_combined <- plot_grid(p_bar, p_main, nrow = 1, rel_widths = c(0.15, 0.85),
                                align = "h", axis = "tb")

        ggsave(out_path, p_combined, width = 12, height = 8, dpi = 300)
        cat(paste0("[SAVE] ", out_path, "\n"))
        return(p_combined)
    }

    # One plot per clustering method
    for (cm in methods) {
        sub_df <- all_data %>% filter(criteria == cm)
        if (nrow(sub_df) == 0) next
        out_path <- file.path(read_dir, paste0("bubble_ARI_", cm, ".png"))
        make_bubble(sub_df, paste0("ARI - ", cm), out_path)
    }

    # Average ARI across all clustering methods
    avg_df <- all_data %>%
        group_by(prefix, trans) %>%
        summarise(ARI = mean(ARI, na.rm = TRUE), .groups = "drop")

    out_path <- file.path(read_dir, "bubble_ARI_average.png")
    make_bubble(avg_df, "ARI - Average across clustering methods", out_path)

    cat("All bubble plots saved!\n")
}