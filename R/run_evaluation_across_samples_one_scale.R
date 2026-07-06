library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(cowplot)

args <- commandArgs(trailingOnly = TRUE)

BASE_DIR <- args[1]
TSV_FILE <- args[2]
sample_index_arg <- args[3]

# ============================================================
# args4:
#   missing:
#       use no_extra_subdir only
#   dropout_0p1:
#       use dropout_0p1 only
#   no_extra_subdir,dropout_0p1,dropout_0p5:
#       combine all these groups into one plot
# ============================================================

if (length(args) >= 4 && !is.na(args[4]) && args[4] != "") {
    extra_subdirs <- str_split(args[4], ",")[[1]]
    extra_subdirs <- str_trim(extra_subdirs)
    extra_subdirs <- extra_subdirs[extra_subdirs != ""]
} else {
    extra_subdirs <- c("no_extra_subdir")
}

# Convert label to real folder path
# no_extra_subdir / none / default means no extra folder
extra_subdir_path_map <- extra_subdirs
extra_subdir_path_map[extra_subdir_path_map %in% c("no_extra_subdir", "none", "default")] <- ""

out_dir_suffix <- paste(extra_subdirs, collapse = "__")

OUT_DIR <- file.path(
    BASE_DIR,
    "Across_sample_bubble_plots",
    out_dir_suffix
)

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

input_types <- c("RNA", "ATAC")
corr_dirs <- c("col", "row")

corr_object_map <- c(
    col = "gene",
    row = "cell"
)

data_modes <- c(
    "default",
    "v1_trans_v2_trans",
    "v1_reverse",
    "v1_trans_v2_trans_norm_100000"
)

mode_suffix_map <- c(
    default = "v1_trans_v2_no_trans",
    v1_trans_v2_trans = "v1_trans_v2_trans",
    v1_reverse = "v1_reverse_v2_no_trans",
    v1_trans_v2_trans_norm_100000 = "v1_trans_v2_trans_norm_100000"
)

# ============================================================
# Get sample names
# ============================================================

sample_table <- read_tsv(TSV_FILE, show_col_types = FALSE)

sample_indices <- str_split(sample_index_arg, ",")[[1]]
sample_indices <- as.integer(sample_indices)

sample_indices <- sample_indices[
    !is.na(sample_indices) &
        sample_indices >= 1 &
        sample_indices <= nrow(sample_table)
]

sample_names <- sample_table[[2]][sample_indices]
sample_names <- sample_names[!is.na(sample_names) & sample_names != ""]

print(sample_names)
print(extra_subdirs)


# Function: read one table_df file
read_one_bubble_table <- function(sample_name, input_type, corr_dir, mode_suffix, extra_subdir = "", extra_subdir_label = "no_extra_subdir") {
    corr_object <- corr_object_map[[corr_dir]]

    sample_dir <- file.path(
        BASE_DIR,
        paste0(sample_name, "_INPUT_", input_type)
    )

    path_parts <- c(
        sample_dir,
        extra_subdir,
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

    path_parts <- path_parts[!is.na(path_parts) & path_parts != ""]

    file_path <- do.call(file.path, as.list(path_parts))

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
            extra_subdir = extra_subdir,
            extra_subdir_label = extra_subdir_label,
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
            extra_subdir_label,
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

            # This is the new x-axis column.
            # Example:
            # no_extra_subdir | col | DCA_mse | pearson
            # dropout_0p1     | row | scVI_mse | spearman
            x_group = paste(
                extra_subdir_label,
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
            extra_subdir_label,
            corr_dir,
            method_combination,
            x_group
        ) %>%
        mutate(
            extra_subdir_label = factor(
                extra_subdir_label,
                levels = extra_subdirs
            ),
            corr_dir = factor(
                corr_dir,
                levels = corr_dirs
            )
        ) %>%
        arrange(
            extra_subdir_label,
            corr_dir,
            method_combination
        ) %>%
        pull(x_group) %>%
        as.character()

    return(x_order)
}

# Function: plot performance bubble plot
plot_performance_bubble <- function(avg_df, title_text, out_png, subtitle_text = NULL) {

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

    plot_df <- avg_df %>%
        mutate(
            trans = factor(trans, levels = rev(perf_order)),
            x_group = factor(x_group, levels = x_order)
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
                size = avg_rank_perf,
                color = avg_mean_performance
            ),
            alpha = 0.9
        ) +
        scale_size_continuous(
            name = "Average rank\n(smaller = better)",
            range = c(12, 1.5)
        ) +
        labs(
            title = title_text,
            subtitle = subtitle_text,
            x = "Dropout group | corr_dir | Method Combination",
            y = "Transformation",
            color = "Average mean performance\n(larger = better)"
        ) +
        theme_minimal(base_size = 12) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            axis.text.y = element_text(size = 10),
            panel.grid.major = element_line(color = "grey90"),
            panel.grid.minor = element_blank(),
            legend.position = "right",
            plot.title = element_text(hjust = 0.5, face = "bold"),
            plot.subtitle = element_text(hjust = 0.5, size = 9)
        )

    p_bar <- ggplot(avg_rank_perf, aes(x = bar_len, y = trans)) +
        geom_bar(stat = "identity", fill = "steelblue", width = 0.7) +
        scale_x_reverse() +
        theme_minimal() +
        theme(
            axis.text.y = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.x = element_blank(),
            axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            panel.grid = element_blank()
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
    ggsave(out_png, p, width = 22, height = 10, dpi = 300)
}


# Function: plot residual bubble plot
plot_residual_bubble <- function(avg_df, title_text, out_png, subtitle_text = NULL) {

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

    plot_df <- avg_df %>%
        mutate(
            trans = factor(trans, levels = rev(resid_order)),
            x_group = factor(x_group, levels = x_order)
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
        ) +
        scale_size_continuous(
            name = "Average rank\n(smaller = better)",
            range = c(12, 1.5)
        ) +
        labs(
            title = title_text,
            subtitle = subtitle_text,
            x = "Dropout group | corr_dir | Method Combination",
            y = "Transformation",
            color = "Average mean residual\n(smaller = better)"
        ) +
        theme_minimal(base_size = 12) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            axis.text.y = element_text(size = 10),
            panel.grid.major = element_line(color = "grey90"),
            panel.grid.minor = element_blank(),
            legend.position = "right",
            plot.title = element_text(hjust = 0.5, face = "bold"),
            plot.subtitle = element_text(hjust = 0.5, size = 9)
        )

    p_bar <- ggplot(avg_rank_resid, aes(x = bar_len, y = trans)) +
        geom_bar(stat = "identity", fill = "coral", width = 0.7) +
        scale_x_reverse() +
        theme_minimal() +
        theme(
            axis.text.y = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.x = element_blank(),
            axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            panel.grid = element_blank()
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
    ggsave(out_png, p, width = 22, height = 10, dpi = 300)
}


# Main
all_aggregated <- list()

for (input_type in input_types) {
    for (data_mode in data_modes) {

        mode_suffix <- mode_suffix_map[[data_mode]]

        message("========================================")
        message("Processing combined plot: ", input_type, " | ", mode_suffix)
        message("Extra subdir groups: ", paste(extra_subdirs, collapse = ", "))
        message("corr_dir groups: ", paste(corr_dirs, collapse = ", "))
        message("========================================")

        df_list_all <- list()

        for (extra_i in seq_along(extra_subdirs)) {

            extra_subdir_label <- extra_subdirs[extra_i]
            extra_subdir_path <- extra_subdir_path_map[extra_i]

            for (corr_dir in corr_dirs) {

                df_list <- lapply(
                    sample_names,
                    read_one_bubble_table,
                    input_type = input_type,
                    corr_dir = corr_dir,
                    mode_suffix = mode_suffix,
                    extra_subdir = extra_subdir_path,
                    extra_subdir_label = extra_subdir_label
                )

                df_list_all <- c(df_list_all, df_list)
            }
        }

        all_df <- bind_rows(df_list_all)

        if (nrow(all_df) == 0) {
            message("[Skip] No files found for: ", input_type, " | ", mode_suffix)
            next
        }

        samples_used <- paste(sort(unique(all_df$sample_name)), collapse = ", ")
        dropout_used <- paste(unique(all_df$extra_subdir_label), collapse = ", ")
        corr_used <- paste(sort(unique(all_df$corr_dir)), collapse = ", ")

        subtitle_text <- str_wrap(
            paste0(
                "Samples included: ", samples_used,
                " | Dropout groups: ", dropout_used,
                " | corr_dir: ", corr_used
            ),
            width = 150
        )

        avg_df <- aggregate_across_samples(all_df)

        sub_out_dir <- file.path(
            OUT_DIR,
            input_type,
            mode_suffix
        )

        dir.create(sub_out_dir, recursive = TRUE, showWarnings = FALSE)

        raw_out_csv <- file.path(
            sub_out_dir,
            paste0(
                "bubble_plot_summary_col_row_",
                mode_suffix,
                "_all_dropout_groups_across_samples_raw_table_df.csv"
            )
        )

        avg_out_csv <- file.path(
            sub_out_dir,
            paste0(
                "bubble_plot_summary_col_row_",
                mode_suffix,
                "_all_dropout_groups_across_samples_avg_rank_table_df.csv"
            )
        )

        perf_out_png <- file.path(
            sub_out_dir,
            paste0(
                "bubble_plot_summary_col_row_",
                mode_suffix,
                "_all_dropout_groups_across_samples_avg_rank_performance.png"
            )
        )

        resid_out_png <- file.path(
            sub_out_dir,
            paste0(
                "bubble_plot_summary_col_row_",
                mode_suffix,
                "_all_dropout_groups_across_samples_avg_rank_residual.png"
            )
        )

        write_csv(all_df, raw_out_csv)
        write_csv(avg_df, avg_out_csv)

        plot_performance_bubble(
            avg_df,
            title_text = paste0(
                "Average Performance across Samples: ",
                input_type,
                " | col + row | ",
                mode_suffix,
                " | all dropout groups"
            ),
            out_png = perf_out_png,
            subtitle_text = subtitle_text
        )

        plot_residual_bubble(
            avg_df,
            title_text = paste0(
                "Average Residual across Samples: ",
                input_type,
                " | col + row | ",
                mode_suffix,
                " | all dropout groups"
            ),
            out_png = resid_out_png,
            subtitle_text = subtitle_text
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
    "bubble_plot_summary_all_input_types_all_modes_all_dropout_groups_across_samples_avg_rank_table_df.csv"
)

write_csv(combined_avg_df, combined_out_csv)

message("Done.")
message("Combined summary saved to: ", combined_out_csv)