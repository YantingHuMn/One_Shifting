library(dplyr)
library(readr)
library(purrr)
library(stringr)

base_dirs <- c(
    HCA = "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA",
    ENCODE = "/dcs07/hongkai/data/yhu1/One_Shifting_Results/ENCODE",
    `10x` = "/dcs10/hongkai/data/yhu1/One_Shifting_Results/10x"
)

table_name <- paste0(
    "bubble_plot_summary_col_gene_",
    "v1_reverse_v2_no_trans_table_df.csv"
)

output_file <- file.path(
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results",
    "col_reverse_representative_sample_ranking.csv"
)

# 1. find all *_INPUT_RNA sample folders
file_info <- imap_dfr(
    base_dirs,
    function(base_dir, dataset) {
        sample_dirs <- list.dirs(
            base_dir,
            recursive = FALSE,
            full.names = TRUE
        )

        sample_dirs <- sample_dirs[
            str_detect(basename(sample_dirs), "_INPUT_RNA$")
        ]

        table_files <- file.path(sample_dirs, table_name)

        tibble(
            dataset = dataset,
            input_folder = basename(sample_dirs),
            file = table_files,
            file_exists = file.exists(table_files)
        )
    }
)

cat("INPUT_RNA folders found:", nrow(file_info), "\n")
cat("Tables found:", sum(file_info$file_exists), "\n")
cat("Tables missing:", sum(!file_info$file_exists), "\n\n")

if (any(!file_info$file_exists)) {
    cat("Missing tables:\n")
    print(file_info %>% filter(!file_exists))
}

file_info <- file_info %>%
    filter(file_exists)

if (nrow(file_info) == 0) {
    stop("No col_reverse table_df files were found.")
}

# 2. read each df
all_results <- pmap_dfr(
    file_info[, c("file", "input_folder", "dataset")],
    function(file, input_folder, dataset) {

        current_df <- read_csv(
            file,
            show_col_types = FALSE
        )

        required_columns <- c(
            "method",
            "data_type",
            "metric",
            "trans",
            "mean_performance",
            "mean_residual"
        )

        missing_columns <- setdiff(
            required_columns,
            colnames(current_df)
        )

        if (length(missing_columns) > 0) {
            stop(
                "\nMissing columns in:\n",
                file,
                "\nColumns: ",
                paste(missing_columns, collapse = ", ")
            )
        }

        current_df %>%
            mutate(
                dataset = dataset,
                input_folder = input_folder,
                source_file = file,
                mean_performance = as.numeric(mean_performance),
                mean_residual = as.numeric(mean_residual)
            )
    }
)

invalid_rows <- all_results %>%
    filter(
        !is.finite(mean_performance) |
        !is.finite(mean_residual)
    )

if (nrow(invalid_rows) > 0) {
    warning(
        nrow(invalid_rows),
        " rows with NA/NaN/Inf were excluded."
    )
}

all_results_valid <- all_results %>%
    filter(
        is.finite(mean_performance),
        is.finite(mean_residual)
    )


sample_counts <- all_results_valid %>%
    group_by(dataset, input_folder, data_type) %>%
    summarise(
        number_of_rows = n(),
        number_of_comparisons = n_distinct(
            interaction(method, metric, trans, drop = TRUE)
        ),
        number_of_methods = n_distinct(method),
        number_of_transformations = n_distinct(trans),
        number_of_metrics = n_distinct(metric),
        .groups = "drop"
    )

cat("\nComparison counts by sample:\n")
print(sample_counts, n = Inf)

unexpected_counts <- sample_counts %>%
    filter(number_of_comparisons != 48)

if (nrow(unexpected_counts) > 0) {
    warning(
        "Some samples do not have 48 method × metric × ",
        "transformation comparisons. See printed table."
    )
}


sample_summary <- all_results_valid %>%
    group_by(dataset, input_folder, data_type) %>%
    summarise(
        mean_performance = mean(
            mean_performance,
            na.rm = TRUE
        ),
        mean_residual = mean(
            mean_residual,
            na.rm = TRUE
        ),
        number_of_comparisons = n_distinct(
            interaction(method, metric, trans, drop = TRUE)
        ),
        .groups = "drop"
    )

# 3. rank samples based on mean_performance and mean_residual
sample_ranking <- sample_summary %>%
    mutate(
        mean_performance_rank = rank(
            -mean_performance,
            ties.method = "average",
            na.last = "keep"
        ),
        mean_residual_rank = rank(
            mean_residual,
            ties.method = "average",
            na.last = "keep"
        ),
        overall_rank_score = (
            mean_performance_rank +
            mean_residual_rank
        ) / 2
    ) %>%
    arrange(
        overall_rank_score,
        desc(mean_performance),
        mean_residual
    ) %>%
    mutate(final_rank = row_number()) %>%
    select(
        final_rank,
        dataset,
        input_folder,
        data_type,
        mean_performance,
        mean_residual,
        mean_performance_rank,
        mean_residual_rank,
        overall_rank_score,
        number_of_comparisons
    )

median_position <- (nrow(sample_ranking) + 1) / 2

representative_sample <- sample_ranking %>%
    mutate(
        distance_from_median_rank =
            abs(final_rank - median_position)
    ) %>%
    slice_min(
        order_by = distance_from_median_rank,
        n = 1,
        with_ties = FALSE
    )

write_csv(sample_ranking, output_file)

cat("\nFinal sample ranking:\n")
print(sample_ranking, n = Inf)

cat("\nMedian rank position:", median_position, "\n")
cat("\nRepresentative sample:\n")
print(representative_sample)

cat("\nSaved to:\n", output_file, "\n")