library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(cowplot)

args <- commandArgs(trailingOnly = TRUE)

BASE_DIR <- args[1]
TSV_FILES <- str_split(args[2], ",")[[1]]
TSV_FILES <- str_trim(TSV_FILES)
TSV_FILES <- TSV_FILES[TSV_FILES != ""]
sample_index_arg <- args[3]

if (length(args) >= 4 && !is.na(args[4]) && args[4] != "") {
    corr_dirs <- str_split(args[4], ",")[[1]]
    corr_dirs <- str_trim(corr_dirs)
    corr_dirs <- corr_dirs[corr_dirs %in% c("col", "row")]

    if (length(corr_dirs) == 0) {
        stop("args4 must contain 'col', 'row', or 'col,row'.")
    }
} else {
    corr_dirs <- c("col", "row")
}

if (length(args) >= 5 && !is.na(args[5]) && args[5] != "") {
    OUT_DIR <- args[5]
} else {
    stop("args5 must be the output directory.")
}

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

input_types <- c("RNA", "ATAC")

corr_object_map <- c(
    col = "gene",
    row = "cell"
)

data_modes <- c("v1_reverse")

mode_suffix_map <- c(
    v1_reverse = "v1_reverse_v2_no_trans"
)

# ============================================================
# Get sample names
# ============================================================

sample_info <- bind_rows(lapply(TSV_FILES, function(TSV_FILE) {
    sample_table <- read_tsv(TSV_FILE, show_col_types = FALSE)

    if (tolower(sample_index_arg) == "all") {
        sample_indices <- seq_len(nrow(sample_table))
    } else {
        sample_indices <- str_split(sample_index_arg, ",")[[1]]
        sample_indices <- as.integer(sample_indices)

        sample_indices <- sample_indices[
            !is.na(sample_indices) &
                sample_indices >= 1 &
                sample_indices <= nrow(sample_table)
        ]
    }

    if (length(TSV_FILES) == 1) {
        sample_base_dir <- BASE_DIR
    } else {
        sample_base_dir <- str_remove(TSV_FILE, "_input_sample_pairs\\.tsv$")
    }

    tibble(
        sample_name = sample_table[[2]][sample_indices],
        sample_base_dir = sample_base_dir
    )
}))

sample_info <- sample_info %>%
    filter(!is.na(sample_name) & sample_name != "")

sample_names <- sample_info$sample_name

print(sample_names)


# Function: read one table_df file
read_one_bubble_table <- function(sample_name, sample_base_dir, input_type, corr_dir, mode_suffix) {
    corr_object <- corr_object_map[[corr_dir]]

    sample_dir <- file.path(
        sample_base_dir,
        paste0(sample_name, "_INPUT_", input_type)
    )

    file_path <- file.path(
        sample_dir,
        paste0(
            "bubble_plot_summary_",
            corr_dir,
            "_",
            corr_object,
            "_",
            mode_suffix,
            "_table_df.csv"
        )
    )

    if (!file.exists(file_path)) {
        message("[Missing] ", file_path)
        return(NULL)
    }

    df <- read_csv(file_path, show_col_types = FALSE)

    df <- df %>%
        mutate(
            sample_name = sample_name,
            input_type = input_type,
            corr_dir = corr_dir,
            mode_suffix = mode_suffix,
            source_file = file_path
        )

    return(df)
}

# Function: aggregate across samples
aggregate_across_samples <- function(all_df) {

    avg_df <- all_df %>%
        group_by(
            input_type,
            corr_dir,
            mode_suffix,
            method,
            metric,
            trans
        ) %>%
        summarise(
            avg_mean_performance = mean(mean_performance, na.rm = TRUE),
            avg_mean_residual = mean(mean_residual, na.rm = TRUE),

            avg_rank_perf = mean(rank_perf, na.rm = TRUE),
            avg_rank_residual = mean(rank_residual, na.rm = TRUE),
            avg_combined_rank = mean(combined_rank, na.rm = TRUE),

            sd_rank_perf = sd(rank_perf, na.rm = TRUE),
            sd_rank_residual = sd(rank_residual, na.rm = TRUE),
            sd_combined_rank = sd(combined_rank, na.rm = TRUE),

            n_sample = n_distinct(sample_name),
            samples = paste(sort(unique(sample_name)), collapse = ";"),
            .groups = "drop"
        ) %>%
        mutate(
            method_combination = paste(method, metric, sep = " | "),

            x_group = paste(
                corr_dir,
                method_combination,
                sep = " | "
            ),

            # keep these columns, although plotting below uses avg_rank directly
            rank_score_perf = max(avg_rank_perf, na.rm = TRUE) + 1 - avg_rank_perf,
            rank_score_residual = max(avg_rank_residual, na.rm = TRUE) + 1 - avg_rank_residual,
            rank_score_combined = max(avg_combined_rank, na.rm = TRUE) + 1 - avg_combined_rank
        )

    return(avg_df)
}

# Function: make x-axis order
make_x_order <- function(avg_df) {

    x_order <- avg_df %>%
        distinct(
            corr_dir,
            method,
            metric,
            method_combination,
            x_group
        ) %>%
        mutate(
            corr_dir = factor(
                corr_dir,
                levels = corr_dirs
            ),
            metric_order = factor(
                tolower(metric),
                levels = c("pearson", "spearman")
            ),
            method_order = factor(
                method,
                levels = c("VAE", "DCA_mse", "scVI_mse", "Transformer_denoise")
            )
        ) %>%
        arrange(
            corr_dir,
            metric_order,
            method_order
        ) %>%
        pull(x_group) %>%
        as.character()

    return(x_order)
}

# Function: plot performance bubble plot
plot_performance_bubble <- function(avg_df, title_text, out_png) {

    # Average rank across all x groups for each transformation
    # smaller avg_rank = better
    avg_rank_perf <- avg_df %>%
        group_by(trans) %>%
        summarise(
            avg_rank = mean(avg_rank_perf, na.rm = TRUE),
            .groups = "drop"
        )

    # Transformation order: best average rank first
    perf_order <- avg_rank_perf %>%
        arrange(avg_rank) %>%
        pull(trans) %>%
        as.character()

    # Put best transformation on top of y-axis
    avg_rank_perf <- avg_rank_perf %>%
        mutate(
            trans = factor(trans, levels = rev(perf_order)),
            bar_len = max(avg_rank, na.rm = TRUE) - avg_rank + min(avg_rank, na.rm = TRUE)
        )

    x_order <- make_x_order(avg_df)

    x_labels <- avg_df %>%
        distinct(x_group, method) %>%
        { setNames(.$method, .$x_group) }

    plot_df <- avg_df %>%
        mutate(
            trans = factor(trans, levels = rev(perf_order)),
            x_group = factor(x_group, levels = x_order),
            metric = factor(tolower(metric), levels = c("pearson", "spearman"), labels = c("Pearson", "Spearman"))
        )

    p_main <- ggplot(
        plot_df,
        aes(
            x = x_group,
            y = trans
        )
    ) + geom_point(
            aes(
                size = avg_rank_perf,
                color = avg_mean_performance
            ),
            alpha = 0.9
        ) + scale_color_gradient(
                low = "#FAD7B1",
                high = "#E76F51"
            ) + scale_size_continuous(
                name = "Average rank",
                range = c(12, 1.5),
                breaks = function(x) seq(min(x), max(x), length.out = 5),
                labels = function(x) rep("", length(x))
            ) + scale_x_discrete(
                labels = x_labels,
                expand = expansion(add = 0.5)
            ) + facet_grid(
                . ~ metric,
                scales = "free_x",
                space = "free_x"
            ) + guides(
                color = guide_colorbar(
                    order = 1,
                    barheight = unit(0.7, "in"),
                    barwidth = unit(0.18, "in"),
                    title.position = "top"
                ),
                size = guide_legend(
                    order = 2,
                    keyheight = unit(0.18, "in"),
                    keywidth = unit(0.18, "in"),
                    override.aes = list(size = c(5, 4, 3, 2, 1))
                )
            ) + labs(
                title = title_text,
                x = NULL,
                y = NULL,
                color = "Average mean\nperformance"
            ) + theme_minimal(base_size = 12) +
                theme(
                    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
                    axis.text.y = element_text(size = 10),
                    panel.grid = element_blank(),
                    panel.spacing.x = unit(0, "lines"),
                    strip.background = element_rect(fill = "grey20", color = "black"),
                    strip.text = element_text(color = "white", face = "bold", size = 10),
                    legend.position = "right",
                    legend.box = "vertical",
                    legend.title = element_text(size = 8),
                    legend.text = element_text(size = 7),
                    legend.spacing.y = unit(0.05, "lines"),
                    legend.box.spacing = unit(0.1, "lines"),
                    plot.title = element_text(hjust = 0.5, face = "bold")
                )

    p_bar <- ggplot(
        avg_rank_perf %>% mutate(
            section = "Transformation",
            shift_group = if_else(
                as.character(trans) %in% c("log2(count+2)", "sqrt+1", "count+1"),
                "One-shifting",
                "Original"
            )
        ),
        aes(x = bar_len, y = trans, fill = shift_group)
    ) +
        geom_bar(stat = "identity", width = 0.7) +
        scale_fill_manual(
            values = c("One-shifting" = "#F6B3B3", "Original" = "#A7D7A1"),
            breaks = c("One-shifting", "Original"),
            labels = c("One-shifting", "Original"),
            name = NULL,
            guide = guide_legend(
                ncol = 1,
                keywidth = unit(0.18, "in"),
                keyheight = unit(0.18, "in")
            )
        ) +
        scale_x_reverse() +
        facet_grid(. ~ section) +
        theme_minimal() +
        theme(
            axis.text.y = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.x = element_blank(),
            axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            panel.grid = element_blank(),
            strip.background = element_rect(fill = "grey20", color = "black"),
            strip.text = element_text(color = "white", face = "bold", size = 10),
            legend.position = "bottom",
            legend.direction = "vertical",
            legend.justification = "left",
            legend.text = element_text(size = 7),
            legend.margin = margin(0, 0, 0, 0)
        )

    p <- plot_grid(
        p_bar,
        p_main,
        nrow = 1,
        rel_widths = c(0.15, 0.85),
        align = "h",
        axis = "tb"
    )

    # wider plot because x-axis can have 24 columns
    n_x_group <- n_distinct(plot_df$x_group)
    plot_width <- max(10, 4 + 0.5 * n_x_group)

    n_x_group <- n_distinct(plot_df$x_group)
    plot_width <- max(8, 3 + 0.4 * n_x_group)

    ggsave(out_png, p, width = plot_width, height = 4, dpi = 300)
}


# Function: plot residual bubble plot
plot_residual_bubble <- function(avg_df, title_text, out_png) {

    # Average rank across all x groups for each transformation
    # smaller avg_rank = better
    avg_rank_resid <- avg_df %>%
        group_by(trans) %>%
        summarise(
            avg_rank = mean(avg_rank_residual, na.rm = TRUE),
            .groups = "drop"
        )

    # Transformation order: best average rank first
    resid_order <- avg_rank_resid %>%
        arrange(avg_rank) %>%
        pull(trans) %>%
        as.character()

    # Put best transformation on top of y-axis
    avg_rank_resid <- avg_rank_resid %>%
        mutate(
            trans = factor(trans, levels = rev(resid_order)),
            bar_len = max(avg_rank, na.rm = TRUE) - avg_rank + min(avg_rank, na.rm = TRUE)
        )

    x_order <- make_x_order(avg_df)

    x_labels <- avg_df %>%
        distinct(x_group, method) %>%
        { setNames(.$method, .$x_group) }

    plot_df <- avg_df %>%
        mutate(
            trans = factor(trans, levels = rev(resid_order)),
            x_group = factor(x_group, levels = x_order),
            metric = factor(tolower(metric), levels = c("pearson", "spearman"), labels = c("Pearson", "Spearman"))
        )

    p_main <- ggplot(
        plot_df,
        aes(
            x = x_group,
            y = trans
        )
    ) +
        geom_point(
            aes(
                size = avg_rank_residual,
                color = avg_mean_residual
            ),
            alpha = 0.9
        ) + scale_color_gradient(
            low = "#4E79B7",
            high = "#E6F0FF"
        ) + scale_size_continuous(
            name = "Average rank",
            range = c(12, 1.5),
            breaks = function(x) seq(min(x), max(x), length.out = 5),
            labels = function(x) rep("", length(x))
        ) +
        scale_x_discrete(
            labels = x_labels,
            expand = expansion(add = 0.5)
        ) +
        facet_grid(
            . ~ metric,
            scales = "free_x",
            space = "free_x"
        ) +
        guides(
            color = guide_colorbar(
                order = 1,
                barheight = unit(0.7, "in"),
                barwidth = unit(0.18, "in"),
                title.position = "top"
            ),
            size = guide_legend(
                order = 2,
                keyheight = unit(0.18, "in"),
                keywidth = unit(0.18, "in"),
                override.aes = list(size = c(5, 4, 3, 2, 1))
            )
        ) +
        labs(
            title = title_text,
            x = NULL,
            y = NULL,
            color = "Average mean\nresidual"
        ) +
        theme_minimal(base_size = 12) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            axis.text.y = element_text(size = 10),
            panel.grid = element_blank(),
            panel.spacing.x = unit(0, "lines"),
            strip.background = element_rect(fill = "grey20", color = "black"),
            strip.text = element_text(color = "white", face = "bold", size = 10),
            legend.position = "right",
            legend.box = "vertical",
            legend.title = element_text(size = 8),
            legend.text = element_text(size = 7),
            legend.spacing.y = unit(0.05, "lines"),
            legend.box.spacing = unit(0.1, "lines"),
            plot.title = element_text(hjust = 0.5, face = "bold")
        )

    p_bar <- ggplot(
        avg_rank_resid %>% mutate(
            section = "Transformation",
            shift_group = if_else(
                as.character(trans) %in% c("log2(count+2)", "sqrt+1", "count+1"),
                "One-shifting",
                "Original"
            )
        ),
        aes(x = bar_len, y = trans, fill = shift_group)
    ) +
        geom_bar(stat = "identity", width = 0.7) +
        scale_fill_manual(
            values = c("One-shifting" = "#F6B3B3", "Original" = "#A7D7A1"),
            breaks = c("One-shifting", "Original"),
            labels = c("One-shifting", "Original"),
            name = NULL,
            guide = guide_legend(
                ncol = 1,
                keywidth = unit(0.18, "in"),
                keyheight = unit(0.18, "in")
            )
        ) +
        scale_x_reverse() +
        facet_grid(. ~ section) +
        theme_minimal() +
        theme(
            axis.text.y = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.x = element_blank(),
            axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            panel.grid = element_blank(),
            strip.background = element_rect(fill = "grey20", color = "black"),
            strip.text = element_text(color = "white", face = "bold", size = 10),
            legend.position = "bottom",
            legend.direction = "vertical",
            legend.justification = "left",
            legend.text = element_text(size = 7),
            legend.margin = margin(0, 0, 0, 0)
        )

    p <- plot_grid(
        p_bar,
        p_main,
        nrow = 1,
        rel_widths = c(0.15, 0.85),
        align = "h",
        axis = "tb"
    )

    # wider plot because x-axis can have 24 columns
    n_x_group <- n_distinct(plot_df$x_group)
    plot_width <- max(10, 4 + 0.5 * n_x_group)

    n_x_group <- n_distinct(plot_df$x_group)
    plot_width <- max(8, 3 + 0.4 * n_x_group)

    ggsave(out_png, p, width = plot_width, height = 4, dpi = 300)
}


# Main
all_aggregated <- list()

for (input_type in input_types) {
    for (data_mode in data_modes) {

        mode_suffix <- mode_suffix_map[[data_mode]]

        message("========================================")
        message("Processing combined plot: ", input_type, " | ", mode_suffix)
        message("corr_dir groups: ", paste(corr_dirs, collapse = ", "))
        message("========================================")

        df_list_all <- list()

        for (corr_dir in corr_dirs) {

            df_list <- lapply(seq_len(nrow(sample_info)), function(i) {
                read_one_bubble_table(
                    sample_name = sample_info$sample_name[i],
                    sample_base_dir = sample_info$sample_base_dir[i],
                    input_type = input_type,
                    corr_dir = corr_dir,
                    mode_suffix = mode_suffix
                )
            })

            df_list_all <- c(df_list_all, df_list)
        }

        all_df <- bind_rows(df_list_all)

        if (nrow(all_df) == 0) {
            message("[Skip] No files found for: ", input_type, " | ", mode_suffix)
            next
        }

        avg_df <- aggregate_across_samples(all_df)

        raw_out_csv <- file.path(
            OUT_DIR,
            paste0(
                "bubble_plot_summary_", input_type, "_col_row_",
                mode_suffix,
                "_across_samples_raw_table_df.csv"
            )
        )

        avg_out_csv <- file.path(
            OUT_DIR,
            paste0(
                "bubble_plot_summary_", input_type, "_col_row_",
                mode_suffix,
                "_across_samples_avg_rank_table_df.csv"
            )
        )

        perf_out_png <- file.path(
            OUT_DIR,
            paste0(
                "bubble_plot_summary_", input_type, "_col_row_",
                mode_suffix,
                "_across_samples_avg_rank_performance.png"
            )
        )

        resid_out_png <- file.path(
            OUT_DIR,
            paste0(
                "bubble_plot_summary_", input_type, "_col_row_",
                mode_suffix,
                "_across_samples_avg_rank_residual.png"
            )
        )

        write_csv(all_df, raw_out_csv)
        write_csv(avg_df, avg_out_csv)

        plot_performance_bubble(
            avg_df,
            title_text = paste0(
                "Average Performance across Samples: ", input_type
            ),
            out_png = perf_out_png
        )

        plot_residual_bubble(
            avg_df,
            title_text = paste0(
                "Average Residual across Samples: ", input_type
            ),
            out_png = resid_out_png
        )

        all_aggregated[[paste(input_type, mode_suffix, sep = "__")]] <- avg_df

        message("[Saved] ", raw_out_csv)
        message("[Saved] ", avg_out_csv)
        message("[Saved] ", perf_out_png)
        message("[Saved] ", resid_out_png)
    }
}


# Save one combined summary table for all input types and modes
combined_avg_df <- bind_rows(all_aggregated)

combined_out_csv <- file.path(
    OUT_DIR,
    "bubble_plot_summary_all_input_types_v1_reverse_v2_no_trans_across_samples_avg_rank_table_df.csv"
)

write_csv(combined_avg_df, combined_out_csv)

message("Done.")
message("Combined summary saved to: ", combined_out_csv)