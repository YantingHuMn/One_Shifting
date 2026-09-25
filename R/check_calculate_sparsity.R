library(arrow)
library(dplyr)
library(readr)
library(parallel)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
    stop("Usage: Rscript calc_sparsity.R file1.tsv file2.tsv file3.tsv output.csv")
}

TSV_FILES <- args[-length(args)]
OUT_FILE <- args[length(args)]
N_CORES <- as.integer(Sys.getenv("N_CORES", "4"))

calc_sparsity <- function(task) {
    
    if (!file.exists(task$feather_path)) {
        message("[Missing] ", task$feather_path)
        return(NULL)
    }
    
    message(
        "Calculating: ",
        task$dataset, " | ",
        task$sample_name, " | ",
        task$data_type
    )
    
    df <- read_feather(task$feather_path)
    
    if (ncol(df) <= 1) {
        message("[Bad file] only one or zero columns: ", task$feather_path)
        return(NULL)
    }
    
    # Remove the first position column
    mat_df <- df[, -1, drop = FALSE]
    
    matrix_entries <- nrow(mat_df) * ncol(mat_df)
    
    na_entries <- sum(vapply(
        mat_df,
        function(x) sum(is.na(x)),
        numeric(1)
    ))
    
    zero_entries <- sum(vapply(
        mat_df,
        function(x) sum(x == 0, na.rm = TRUE),
        numeric(1)
    ))
    
    non_na_entries <- matrix_entries - na_entries
    
    sparsity <- if (non_na_entries > 0) {
        zero_entries / non_na_entries
    } else {
        NA_real_
    }
    
    data.frame(
        dataset = task$dataset,
        sample_name = task$sample_name,
        data_type = task$data_type,
        sparsity = sparsity,
        n_row = nrow(mat_df),
        n_col = ncol(mat_df),
        matrix_entries = matrix_entries,
        non_na_entries = non_na_entries,
        zero_entries = zero_entries,
        na_entries = na_entries,
        feather_path = task$feather_path
    )
}

task_list <- list()

for (tsv_file in TSV_FILES) {
    
    if (!file.exists(tsv_file)) {
        message("[Missing TSV] ", tsv_file)
        next
    }
    
    pair_df <- read_tsv(
        tsv_file,
        col_types = cols(.default = col_character())
    )
    
    if (!"sample_name" %in% names(pair_df)) {
        stop("TSV must contain the sample_name column: ", tsv_file)
    }
    
    dataset <- sub(
        "_input_sample_pairs\\.tsv$",
        "",
        basename(tsv_file),
        ignore.case = TRUE
    )
    
    result_base_dir <- dirname(normalizePath(tsv_file))
    
    sample_names <- unique(pair_df$sample_name)
    sample_names <- sample_names[!is.na(sample_names) & sample_names != ""]
    
    for (sample_name in sample_names) {
        for (data_type in c("RNA", "ATAC")) {
            
            input_folder <- paste0(
                sample_name,
                "_INPUT_",
                data_type
            )
            
            target_folder <- paste0(
                sample_name,
                "_",
                data_type
            )
            
            feather_path <- file.path(
                result_base_dir,
                dataset,
                input_folder,
                target_folder,
                "Count_Matrix_norm_by_no_norm.feather"
            )
            
            task_list[[length(task_list) + 1]] <- list(
                dataset = dataset,
                sample_name = sample_name,
                data_type = data_type,
                feather_path = feather_path
            )
        }
    }
}

if (length(task_list) == 0) {
    stop("No samples were found in the TSV files.")
}

task_key <- vapply(
    task_list,
    function(x) paste(x$dataset, x$feather_path, sep = "__"),
    character(1)
)

task_list <- task_list[!duplicated(task_key)]

n_workers <- min(N_CORES, length(task_list))

result_list <- mclapply(
    task_list,
    calc_sparsity,
    mc.cores = n_workers
)

sparsity_df <- bind_rows(result_list) %>%
    arrange(dataset, sample_name, data_type)

dir.create(
    dirname(OUT_FILE),
    recursive = TRUE,
    showWarnings = FALSE
)

write_csv(sparsity_df, OUT_FILE)

message("========================================")
message("Sparsity results:")
print(sparsity_df)

message("========================================")
message("Saved:")
message(OUT_FILE)