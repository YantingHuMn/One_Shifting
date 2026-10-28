
library(arrow)
library(dplyr)
library(readr)
library(parallel)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
    stop("Usage: Rscript calc_sparsity.R sample_name read_dir output.csv")
}

sample_name <- args[1]
read_dir <- args[2]
OUT_FILE <- args[3]
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

dataset <- basename(normalizePath(read_dir))

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
        read_dir,
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

if (nrow(sparsity_df) > 0) {
    if (!file.exists(OUT_FILE)) {
        write_csv(sparsity_df, OUT_FILE, na = "")
    } else {
        write.table(
            sparsity_df,
            OUT_FILE,
            sep = ",",
            row.names = FALSE,
            col.names = FALSE,
            append = TRUE,
            na = ""
        )
    }
}

message("========================================")
message("Sparsity results:")
print(sparsity_df)

message("========================================")
message("Saved:")
message(OUT_FILE)
