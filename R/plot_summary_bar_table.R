plot_summary_bar_table <- function(csv_path, filter_method = NULL, filter_trans = NULL) {

    # Load Libraries
    suppressPackageStartupMessages({
        library(ggplot2)
        library(dplyr)
        library(tidyr)
    })


    base_path <- sub("\\.csv$", "", csv_path)

    df <- read.csv(csv_path)


    if (!is.null(filter_method) && filter_method != "") {

        df <- df %>%
            filter(tolower(method) != tolower(filter_method))

        base_path <- paste0(base_path, "_no_", filter_method)
    }


    if (!is.null(filter_trans) && filter_trans != "") {

        df <- df %>%
            filter(tolower(trans) != tolower(filter_trans))

        base_path <- paste0(base_path, "_no_", filter_trans)
    }


    # ============================
    # Performance
    # ============================

    perf_order <- df %>%
        group_by(trans) %>%
        summarise(
            mean_value = mean(mean_performance, na.rm = TRUE)
        ) %>%
        arrange(desc(mean_value)) %>%
        pull(trans)


    df_perf <- df %>%
        mutate(
            trans = factor(
                trans,
                levels = perf_order
            )
        )


    p_perf <- ggplot(
        df_perf,
        aes(
            x = trans,
            y = mean_performance,
            fill = method
        )
    ) +
        geom_bar(
            stat = "identity",
            position = position_dodge(width = 0.8),
            width = 0.6
        ) +
        facet_wrap(
            ~metric,
            nrow = 1,
            scales = "free_y"
        ) +
        theme_bw() +
        theme(
            axis.text.x = element_text(
                angle = 45,
                hjust = 1,
                size = 10
            ),
            axis.text.y = element_text(
                size = 10
            ),
            axis.title = element_text(
                size = 12
            ),
            plot.title = element_text(
                size = 14,
                face = "bold"
            ),
            legend.position = "right",
            legend.text = element_text(
                size = 9
            ),
            legend.title = element_text(
                size = 10
            ),
            strip.text = element_text(
                face = "bold",
                size = 10
            )
        ) +
        coord_cartesian(
            ylim = c(0, 0.18)
        ) +
        labs(
            x = "Transformation",
            y = "Mean Performance",
            title = "Mean Performance Comparison"
        )


    ggsave(
        paste0(base_path, "_performance_bar.png"),
        p_perf,
        width = 6,
        height = 3,
        dpi = 300
    )


    # ============================
    # Residual
    # ============================

    resid_order <- df %>%
        group_by(trans) %>%
        summarise(
            mean_value = mean(mean_residual, na.rm = TRUE)
        ) %>%
        arrange(mean_value) %>%
        pull(trans)


    df_resid <- df %>%
        mutate(
            trans = factor(
                trans,
                levels = resid_order
            )
        )


    p_resid <- ggplot(
        df_resid,
        aes(
            x = trans,
            y = mean_residual,
            fill = method
        )
    ) +
        geom_bar(
            stat = "identity",
            position = position_dodge(width = 0.8),
            width = 0.6
        ) +
        facet_wrap(
            ~metric,
            nrow = 1,
            scales = "free_y"
        ) +
        theme_bw() +
        theme(
            axis.text.x = element_text(
                angle = 45,
                hjust = 1,
                size = 10
            ),
            axis.text.y = element_text(
                size = 10
            ),
            axis.title = element_text(
                size = 12
            ),
            plot.title = element_text(
                size = 14,
                face = "bold"
            ),
            legend.position = "right",
            legend.text = element_text(
                size = 9
            ),
            legend.title = element_text(
                size = 10
            ),
            strip.text = element_text(
                face = "bold",
                size = 10
            )
        ) +
        coord_cartesian(
            ylim = c(0, 0.11)
        ) +
        labs(
            x = "Transformation",
            y = "Mean Residual",
            title = "Mean Residual Comparison"
        )


    ggsave(
        paste0(base_path, "_residual_bar.png"),
        p_resid,
        width = 6,
        height = 3,
        dpi = 300
    )


    cat("Bar plots saved!\n")
}