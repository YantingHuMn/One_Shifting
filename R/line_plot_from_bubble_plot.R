line_plot_from_bubble_plot <- function(df1_path, df2_path, df3_path, name1, name2, name3, trans, out_dir) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(ggplot2)
        library(dplyr)
    })

    df1 <- read.csv(df1_path, stringsAsFactors = FALSE)
    df2 <- read.csv(df2_path, stringsAsFactors = FALSE)
    df3 <- read.csv(df3_path, stringsAsFactors = FALSE)

    df1$source_name <- name1
    df2$source_name <- name2
    df3$source_name <- name3

    df_all <- bind_rows(df1, df2, df3)

    df_trans <- df_all %>%
        filter(trans == !!trans)

    if (nrow(df_trans) == 0) {
        stop(paste("No rows found for trans =", trans))
    }

    df_trans <- df_trans %>%
        mutate(
            source_name = factor(source_name, levels = c(name1, name2, name3)),
            line_group = paste(method, metric, sep = " | ")
        ) %>%
        arrange(source_name)

    # ---- Plot 1: mean_performance ----
    p_perf <- ggplot(
        df_trans,
        aes(
            x = source_name,
            y = avg_mean_performance,
            group = line_group,
            color = line_group
        )
    ) +
        geom_line(linewidth = 1) +
        geom_point(size = 2.5) +
        theme_bw() +
        labs(
            title = paste0("Mean Performance across files (trans = ", trans, ")"),
            x = "Input file / dropout setting",
            y = "mean_performance",
            color = "Method | Metric"
        ) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(hjust = 0.5, face = "bold")
        )

    # ---- Plot 2: mean_residual ----
    p_resid <- ggplot(
        df_trans,
        aes(
            x = source_name,
            y = avg_mean_residual,
            group = line_group,
            color = line_group
        )
    ) +
        geom_line(linewidth = 1) +
        geom_point(size = 2.5) +
        theme_bw() +
        labs(
            title = paste0("Mean Residual across files (trans = ", trans, ")"),
            x = "Input file / dropout setting",
            y = "mean_residual",
            color = "Method | Metric"
        ) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(hjust = 0.5, face = "bold")
        )

    # save path
    # trans_safe <- gsub("[^A-Za-z0-9_]+", "_", trans)
    trans_safe <- trans

    perf_path <- file.path(out_dir, paste0("line_plot_mean_performance_", trans_safe, ".png"))
    resid_path <- file.path(out_dir, paste0("line_plot_mean_residual_", trans_safe, ".png"))

    ggsave(perf_path, p_perf, width = 10, height = 6, dpi = 300)
    ggsave(resid_path, p_resid, width = 10, height = 6, dpi = 300)

    cat("Saved:\n")
    cat(perf_path, "\n")
    cat(resid_path, "\n")

    return(invisible(list(
        filtered_df = df_trans,
        performance_plot = p_perf,
        residual_plot = p_resid,
        performance_path = perf_path,
        residual_path = resid_path
    )))
}

# folder = "v1_reverse_v2_no_trans"

# line_plot_from_bubble_plot(
#     df1_path = paste0("/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/Across_sample_bubble_plots/no_extra_subdir/RNA/col/", folder, "/bubble_plot_summary_col_gene_", folder, "_across_samples_avg_rank_table_df.csv"),
#     df2_path = paste0("/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/Across_sample_bubble_plots/dropout_0p5/RNA/col/", folder, "/bubble_plot_summary_col_gene_", folder, "_across_samples_avg_rank_table_df.csv"),
#     df3_path = paste0("/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/Across_sample_bubble_plots/dropout_0p1/RNA/col/", folder, "/bubble_plot_summary_col_gene_", folder, "_across_samples_avg_rank_table_df.csv"),
#     name1 = "no_dropout",
#     name2 = "dropout0p5",
#     name3 = "dropout0p1",
#     trans = "log2(count+2)",
#     out_dir = paste0("/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/Across_sample_bubble_plots/no_extra_subdir/RNA/col/", folder)
# )