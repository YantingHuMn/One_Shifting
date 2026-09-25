library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 6) {
    stop("Usage: Rscript plot_sparsity_difference.R sample.csv METHOD RNA|ATAC pearson|spearman performance|residual OUT_DIR")
}

CSV_FILE <- args[1]
METHOD <- trimws(strsplit(args[2], ",")[[1]])
MODALITY <- toupper(strsplit(args[3], ",")[[1]])
METRIC <- tolower(strsplit(args[4], ",")[[1]])
VALUE_TYPE <- tolower(strsplit(args[5], ",")[[1]])
OUT_DIR <- args[6]

if (!all(MODALITY %in% c("RNA", "ATAC"))) {
    stop("MODALITY must be RNA or ATAC.")
}

if (!all(METRIC %in% c("pearson", "spearman"))) {
    stop("METRIC must be pearson or spearman.")
}

if (!all(VALUE_TYPE %in% c("performance", "residual"))) {
    stop("VALUE_TYPE must be performance or residual.")
}

value_column <- paste0("mean_", VALUE_TYPE)

difference_names <- c(
    "count_plus_1_minus_no_trans",
    "sqrt_plus_1_minus_sqrt",
    "log2_count_plus_2_minus_log2"
)

difference_columns <- unlist(
    lapply(METHOD, function(current_method) {
        unlist(
            lapply(value_column, function(current_value_column) {
                unlist(
                    lapply(METRIC, function(current_metric) {
                        paste(
                            current_method,
                            current_metric,
                            difference_names,
                            current_value_column,
                            sep = "_"
                        )
                    })
                )
            })
        )
    })
)

method_lookup <- setNames(
    unlist(
        lapply(METHOD, function(current_method) {
            rep(
                current_method,
                length(value_column) *
                    length(METRIC) *
                    length(difference_names)
            )
        })
    ),
    difference_columns
)

df <- read_csv(
    CSV_FILE,
    show_col_types = FALSE
)

required_columns <- c(
    "data_type",
    "sparsity",
    difference_columns
)

if (!all(required_columns %in% names(df))) {
    stop(
        "Missing columns: ",
        paste(
            setdiff(required_columns, names(df)),
            collapse = ", "
        )
    )
}

plot_df <- df %>%
    filter(toupper(data_type) %in% MODALITY) %>%
    select(
        any_of(c("dataset", "sample_name")),
        data_type,
        sparsity,
        all_of(difference_columns)
    ) %>%
    pivot_longer(
        cols = all_of(difference_columns),
        names_to = "difference_type",
        values_to = "difference"
    ) %>%
    mutate(
        transformation_pair = case_when(
            grepl("count_plus_1_minus_no_trans", difference_type) ~
                "count+1 - no transformation",
            grepl("sqrt_plus_1_minus_sqrt", difference_type) ~
                "sqrt(x+1) - sqrt(x)",
            grepl("log2_count_plus_2_minus_log2", difference_type) ~
                "log2(x+2) - log2(x+1)"
        ),
        transformation_pair = factor(
            transformation_pair,
            levels = c(
                "count+1 - no transformation",
                "sqrt(x+1) - sqrt(x)",
                "log2(x+2) - log2(x+1)"
            )
        ),
        method_label = unname(method_lookup[difference_type]),
        method_label = factor(
            method_label,
            levels = METHOD
        ),
        metric_label = case_when(
            grepl("_pearson_", difference_type) ~ "Pearson",
            grepl("_spearman_", difference_type) ~ "Spearman"
        ),
        value_type_label = case_when(
            grepl("_mean_performance$", difference_type) ~ "Performance",
            grepl("_mean_residual$", difference_type) ~ "Residual"
        )
    ) %>%
    filter(
        is.finite(sparsity),
        is.finite(difference)
    )

p <- ggplot(
    plot_df,
    aes(
        x = sparsity,
        y = difference,
        color = transformation_pair
    )
) +
    geom_hline(
        yintercept = 0,
        color = "grey45",
        linewidth = 0.6,
        linetype = "dashed"
    ) +
    geom_point(
        size = 2.8,
        alpha = 0.8
    ) +
    scale_color_manual(
        values = c(
            "log2(x+2) - log2(x+1)" = "#4F587D",
            "sqrt(x+1) - sqrt(x)" = "#C68DC0",
            "count+1 - no transformation" = "#C2E0EE"
        ),
        name = "Transformation pair"
    ) +
    labs(
        title = paste(
            paste(METHOD, collapse = " / "),
            paste(MODALITY, collapse = " / "),
            paste(tools::toTitleCase(METRIC), collapse = " / "),
            paste(tools::toTitleCase(VALUE_TYPE), collapse = " / ")
        ),
        x = "Sparsity",
        y = "Difference (shifted - original)"
    ) +
    theme_classic(base_size = 12) +
    theme(
        plot.title = element_text(
            size = 14,
            face = "bold",
            hjust = 0.5
        ),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 9)
    )

if (length(METHOD) > 1) {
    p <- p +
        facet_grid(
            rows = vars(data_type, value_type_label),
            cols = vars(method_label, metric_label),
            scales = "free_y"
        )
} else if (length(MODALITY) > 1 || length(METRIC) > 1 || length(VALUE_TYPE) > 1) {
    p <- p +
        facet_grid(
            rows = vars(data_type, value_type_label),
            cols = vars(metric_label),
            scales = "free_y"
        )
}

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

png_file <- file.path(
    OUT_DIR,
    paste0(
        "sparsity_vs_difference_",
        paste(METHOD, collapse = "_"),
        "_",
        tolower(paste(MODALITY, collapse = "_")),
        "_",
        paste(METRIC, collapse = "_"),
        "_",
        paste(VALUE_TYPE, collapse = "_"),
        ".png"
    )
)

ggsave(png_file, p, width = 7, height = 5.5, units = "in", dpi = 400, bg = "white")

message("Saved:")
message(png_file)