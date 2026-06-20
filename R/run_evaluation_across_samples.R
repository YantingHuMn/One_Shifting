library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(cowplot)

args <- commandArgs(trailingOnly = TRUE)
BASE_DIR <- args[1] 
TSV_FILE <- args[2]
sample_index_arg <- args[3]
extra_subdir <- ""
if (length(args) >= 4) {
    extra_subdir <- args[4]
}

out_dir_suffix <- ifelse(extra_subdir == "", "no_extra_subdir", extra_subdir)

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

# Get sample names
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

# Function: read one table_df file
read_one_bubble_table <- function(sample_name, input_type, corr_dir, mode_suffix, extra_subdir = "") {
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
            source_file = file_path
        )
    
    return(df)
}

# Function: aggregate across samples for one hierarchy level
aggregate_across_samples <- function(all_df) {
    
    avg_df <- all_df %>%
        group_by(input_type, corr_dir, mode_suffix, method, metric, trans) %>%
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
        ) %>% mutate(
            method_combination = paste(method, metric, sep = " | "),
            
            # keep these columns, although plotting below uses avg_rank directly
            rank_score_perf = max(avg_rank_perf, na.rm = TRUE) + 1 - avg_rank_perf,
            rank_score_residual = max(avg_rank_residual, na.rm = TRUE) + 1 - avg_rank_residual,
            rank_score_combined = max(avg_combined_rank, na.rm = TRUE) + 1 - avg_combined_rank
        )
    
    return(avg_df)
}

# Function: plot performance bubble plot
plot_performance_bubble <- function(avg_df, title_text, out_png, subtitle_text = NULL) {
    
    # Average rank across method combinations for each transformation
    # smaller avg_rank = better
    avg_rank_perf <- avg_df %>%
        group_by(trans) %>%
        summarise(avg_rank = mean(avg_rank_perf, na.rm = TRUE), .groups = "drop")
    
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
    
    # X-axis order follows the best transformation
    # larger avg_mean_performance = better
    best_perf_trans <- perf_order[1]
    
    method_order <- avg_df %>%
        filter(trans == best_perf_trans) %>%
        arrange(desc(avg_mean_performance)) %>%
        pull(method_combination) %>%
        unique()
    
    method_order <- c(
        method_order,
        setdiff(unique(avg_df$method_combination), method_order)
    )
    
    plot_df <- avg_df %>%
        mutate(
            trans = factor(trans, levels = rev(perf_order)),
            method_combination = factor(method_combination, levels = method_order)
        )
    
    p_main <- ggplot(
        plot_df,
        aes(
            x = method_combination,
            y = trans
        )) +
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
            x = "Method Combination",
            y = "Transformation",
            color = "Average mean performance\n(larger = better)"
        ) +
        theme_minimal(base_size = 12) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
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
    
    ggsave(out_png, p, width = 14, height = 10, dpi = 300)
}

# Function: plot residual bubble plot
plot_residual_bubble <- function(avg_df, title_text, out_png, subtitle_text = NULL) {
    
    # Average rank across method combinations for each transformation
    # smaller avg_rank = better
    avg_rank_resid <- avg_df %>%
        group_by(trans) %>%
        summarise(avg_rank = mean(avg_rank_residual, na.rm = TRUE), .groups = "drop")
    
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
    
    # X-axis order follows the best transformation
    # smaller avg_mean_residual = better
    best_resid_trans <- resid_order[1]
    
    method_order <- avg_df %>%
        filter(trans == best_resid_trans) %>%
        arrange(avg_mean_residual) %>%
        pull(method_combination) %>%
        unique()
    
    method_order <- c(
        method_order,
        setdiff(unique(avg_df$method_combination), method_order)
    )
    
    plot_df <- avg_df %>%
        mutate(
            trans = factor(trans, levels = rev(resid_order)),
            method_combination = factor(method_combination, levels = method_order)
        )
    
    p_main <- ggplot(
        plot_df,
        aes(
            x = method_combination,
            y = trans
        )) +
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
            x = "Method Combination",
            y = "Transformation",
            color = "Average mean residual\n(smaller = better)"
        ) +
        theme_minimal(base_size = 12) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
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
    
    ggsave(out_png, p, width = 14, height = 10, dpi = 300)
}

# Main 
all_aggregated <- list()

for (input_type in input_types) {
    for (corr_dir in corr_dirs) {
        for (data_mode in data_modes) {
            
            mode_suffix <- mode_suffix_map[[data_mode]]
            
            message("========================================")
            message("Processing: ", input_type, " | ", corr_dir, " | ", mode_suffix)
            message("========================================")
            
            df_list <- lapply(
                sample_names,
                read_one_bubble_table,
                input_type = input_type,
                corr_dir = corr_dir,
                mode_suffix = mode_suffix,
                extra_subdir = extra_subdir
            )
            
            all_df <- bind_rows(df_list)
            
            if (nrow(all_df) == 0) {
                message("[Skip] No files found for: ", input_type, " | ", corr_dir, " | ", mode_suffix)
                next
            }
            
            samples_used <- paste(sort(unique(all_df$sample_name)), collapse = ", ")
            subtitle_text <- str_wrap(
                paste0("Samples included: ", samples_used),
                width = 120
            )
            
            avg_df <- aggregate_across_samples(all_df)
            corr_object <- corr_object_map[[corr_dir]]
            
            sub_out_dir <- file.path(
                OUT_DIR,
                input_type,
                corr_dir,
                mode_suffix
            )
            dir.create(sub_out_dir, recursive = TRUE, showWarnings = FALSE)
            
            raw_out_csv <- file.path(
                sub_out_dir,
                paste0(
                    "bubble_plot_summary_",
                    corr_dir,
                    "_",
                    corr_object,
                    "_",
                    mode_suffix,
                    "_across_samples_raw_table_df.csv"
                )
            )
            
            avg_out_csv <- file.path(
                sub_out_dir,
                paste0(
                    "bubble_plot_summary_",
                    corr_dir,
                    "_",
                    corr_object,
                    "_",
                    mode_suffix,
                    "_across_samples_avg_rank_table_df.csv"
                )
            )
            
            perf_out_png <- file.path(
                sub_out_dir,
                paste0(
                    "bubble_plot_summary_",
                    corr_dir,
                    "_",
                    corr_object,
                    "_",
                    mode_suffix,
                    "_across_samples_avg_rank_performance.png"
                )
            )
            
            resid_out_png <- file.path(
                sub_out_dir,
                paste0(
                    "bubble_plot_summary_",
                    corr_dir,
                    "_",
                    corr_object,
                    "_",
                    mode_suffix,
                    "_across_samples_avg_rank_residual.png"
                )
            )
            
            write_csv(all_df, raw_out_csv)
            write_csv(avg_df, avg_out_csv)
            
            plot_performance_bubble(
                avg_df,
                title_text = paste0("Average Performance across Samples: ", input_type, " | ", corr_dir, " | ", mode_suffix),
                out_png = perf_out_png,
                subtitle_text = subtitle_text
            )
            
            plot_residual_bubble(
                avg_df,
                title_text = paste0("Average Residual across Samples: ", input_type, " | ", corr_dir, " | ", mode_suffix),
                out_png = resid_out_png,
                subtitle_text = subtitle_text
            )
            
            all_aggregated[[paste(input_type, corr_dir, mode_suffix, sep = "__")]] <- avg_df
            
            message("[Saved] ", avg_out_csv)
            message("[Saved] ", perf_out_png)
            message("[Saved] ", resid_out_png)
        }
    }
}

# Save one combined summary table for all hierarchy levels
combined_avg_df <- bind_rows(all_aggregated)

combined_out_csv <- file.path(
    OUT_DIR,
    "bubble_plot_summary_all_input_types_all_modes_across_samples_avg_rank_table_df.csv"
)

write_csv(combined_avg_df, combined_out_csv)

message("Done.")
message("Combined summary saved to: ", combined_out_csv)