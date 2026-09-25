library(dplyr)
library(readr)
library(parallel)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 5) {
    stop("Usage: Rscript add_performance.R HCA.tsv ENCODE.tsv 10x.tsv sample_sparsity.csv METHOD")
}

TSV_FILES <- args[1:3]
SPARSITY_FILE <- args[4]
METHODS <- trimws(strsplit(args[5], ",", fixed = TRUE)[[1]])
METHODS <- unique(METHODS[METHODS != ""])
N_CORES <- as.integer(Sys.getenv("N_CORES", "4"))

metrics <- c("pearson", "spearman")

transformations <- c(
    "no_trans",
    "sqrt",
    "sqrt+1",
    "log2",
    "count+1",
    "log2(count+2)"
)

transformation_names <- c(
    "no_trans",
    "sqrt",
    "sqrt_plus_1",
    "log2",
    "count_plus_1",
    "log2_count_plus_2"
)

value_columns <- c(
    "mean_performance",
    "mean_residual"
)

base_column_names <- unlist(
    lapply(value_columns, function(value_column) {
        unlist(
            lapply(metrics, function(metric_name) {
                paste(
                    metric_name,
                    transformation_names,
                    value_column,
                    sep = "_"
                )
            })
        )
    })
)

difference_pairs <- list(
    count_plus_1_minus_no_trans = c("count_plus_1", "no_trans"),
    sqrt_plus_1_minus_sqrt = c("sqrt_plus_1", "sqrt"),
    log2_count_plus_2_minus_log2 = c("log2_count_plus_2", "log2")
)

difference_column_names <- unlist(
    lapply(value_columns, function(value_column) {
        unlist(
            lapply(metrics, function(metric_name) {
                paste(
                    metric_name,
                    names(difference_pairs),
                    value_column,
                    sep = "_"
                )
            })
        )
    })
)

single_method_column_names <- c(
    base_column_names,
    difference_column_names
)

new_column_names <- unlist(
    lapply(METHODS, function(method_name) {
        paste(
            method_name,
            single_method_column_names,
            sep = "_"
        )
    }),
    use.names = FALSE
)

dataset_dirs <- list()

for (tsv_file in TSV_FILES) {
    
    if (!file.exists(tsv_file)) {
        stop("Missing TSV: ", tsv_file)
    }
    
    dataset <- sub(
        "_input_sample_pairs\\.tsv$",
        "",
        basename(tsv_file),
        ignore.case = TRUE
    )
    
    dataset_dirs[[dataset]] <- file.path(
        dirname(normalizePath(tsv_file)),
        dataset
    )
}

sparsity_df <- read_csv(
    SPARSITY_FILE,
    show_col_types = FALSE
)

required_columns <- c(
    "dataset",
    "sample_name",
    "data_type"
)

if (!all(required_columns %in% names(sparsity_df))) {
    stop("Sparsity CSV must contain dataset, sample_name, and data_type columns.")
}

sparsity_df <- sparsity_df %>%
    select(-any_of(c(single_method_column_names, new_column_names)))

extract_values <- function(i) {
    
    dataset <- sparsity_df$dataset[i]
    sample_name <- sparsity_df$sample_name[i]
    data_type <- sparsity_df$data_type[i]
    
    result <- setNames(
        rep(NA_real_, length(new_column_names)),
        new_column_names
    )
    
    if (!dataset %in% names(dataset_dirs)) {
        message("[Unknown dataset] ", dataset)
        return(as.data.frame(as.list(result)))
    }
    
    summary_path <- file.path(
        dataset_dirs[[dataset]],
        paste0(sample_name, "_INPUT_", data_type),
        "bubble_plot_summary_col_gene_v1_reverse_v2_no_trans.csv"
    )
    
    if (!file.exists(summary_path)) {
        message("[Missing] ", summary_path)
        return(as.data.frame(as.list(result)))
    }
    
    summary_df <- read_csv(
        summary_path,
        show_col_types = FALSE
    )
    
    for (method_name in METHODS) {
        
        method_df <- summary_df %>%
            filter(method == method_name)
        
        if (nrow(method_df) == 0) {
            message(
                "[Method missing] ",
                method_name,
                " | ",
                dataset,
                " | ",
                sample_name,
                " | ",
                data_type
            )
            
            next
        }
        
        for (value_column in value_columns) {
            for (metric_name in metrics) {
                for (j in seq_along(transformations)) {
                    
                    selected_row <- method_df %>%
                        filter(
                            tolower(metric) == metric_name,
                            trans == transformations[j]
                        )
                    
                    column_name <- paste(
                        method_name,
                        metric_name,
                        transformation_names[j],
                        value_column,
                        sep = "_"
                    )
                    
                    if (nrow(selected_row) > 0) {
                        result[column_name] <- as.numeric(
                            selected_row[[value_column]][1]
                        )
                    }
                }
            }
        }
        
        for (value_column in value_columns) {
            for (metric_name in metrics) {
                for (difference_name in names(difference_pairs)) {
                    
                    shifted_name <- difference_pairs[[difference_name]][1]
                    original_name <- difference_pairs[[difference_name]][2]
                    
                    shifted_column <- paste(
                        method_name,
                        metric_name,
                        shifted_name,
                        value_column,
                        sep = "_"
                    )
                    
                    original_column <- paste(
                        method_name,
                        metric_name,
                        original_name,
                        value_column,
                        sep = "_"
                    )
                    
                    difference_column <- paste(
                        method_name,
                        metric_name,
                        difference_name,
                        value_column,
                        sep = "_"
                    )
                    
                    result[difference_column] <- (
                        result[shifted_column] -
                        result[original_column]
                    )
                }
            }
        }
    }
    
    as.data.frame(as.list(result))
}

n_workers <- min(
    N_CORES,
    nrow(sparsity_df)
)

performance_list <- mclapply(
    seq_len(nrow(sparsity_df)),
    extract_values,
    mc.cores = n_workers
)

performance_df <- bind_rows(performance_list)

output_df <- bind_cols(
    sparsity_df,
    performance_df
)

write_csv(
    output_df,
    SPARSITY_FILE
)

message("========================================")
message("Methods: ", paste(METHODS, collapse = ", "))
message(
    "Added ",
    length(single_method_column_names),
    " columns per method and ",
    length(new_column_names),
    " columns in total to:"
)
message(SPARSITY_FILE)