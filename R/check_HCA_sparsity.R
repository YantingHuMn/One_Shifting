library(arrow)
library(dplyr)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
BASE_DIR <- args[1]

sample_names <- c(
    "PF_specimen",
    "Hrv39",
    "Hrv3",
    "HSB5871",
    "LGS1OD",
    "LGS1OS",
    "HDBR10084192_HDBR10142863",
    "HDBR10084193_HDBR10142864"
)

data_info <- list(
    ATAC = list(
        folder_suffix = "_ATAC",
        feather_name = "activity_counts.feather"
    ),
    RNA = list(
        folder_suffix = "_RNA",
        feather_name = "rna_counts.feather"
    )
)

calc_sparsity <- function(feather_path) {
    
    if (!file.exists(feather_path)) {
        message("[Missing] ", feather_path)
        return(NULL)
    }
    
    df <- read_feather(feather_path)
    
    if (ncol(df) <= 1) {
        message("[Bad file] only one or zero columns: ", feather_path)
        return(NULL)
    }
    
    # remove first column: pos
    mat_df <- df[, -1, drop = FALSE]
    
    total_entries <- nrow(mat_df) * ncol(mat_df)
    zero_entries <- sum(mat_df == 0, na.rm = TRUE)
    na_entries <- sum(is.na(mat_df))
    
    sparsity <- zero_entries / total_entries
    
    out <- data.frame(
        n_row = nrow(mat_df),
        n_col = ncol(mat_df),
        total_entries = total_entries,
        zero_entries = zero_entries,
        na_entries = na_entries,
        sparsity = sparsity
    )
    
    return(out)
}

result_list <- list()

for (sample_name in sample_names) {
    for (data_type in names(data_info)) {
        
        folder_path <- file.path(
            BASE_DIR,
            paste0(sample_name, data_info[[data_type]]$folder_suffix)
        )
        
        feather_path <- file.path(
            folder_path,
            data_info[[data_type]]$feather_name
        )
        
        message("Checking: ", sample_name, " | ", data_type)
        message("File: ", feather_path)
        
        res <- calc_sparsity(feather_path)
        
        if (is.null(res)) {
            next
        }
        
        res <- res %>%
            mutate(
                sample_name = sample_name,
                data_type = data_type,
                feather_path = feather_path,
                .before = 1
            )
        
        result_list[[paste(sample_name, data_type, sep = "__")]] <- res
    }
}

sparsity_df <- bind_rows(result_list)

summary_df <- sparsity_df %>%
    group_by(data_type) %>%
    summarise(
        n_sample = n_distinct(sample_name),
        avg_sparsity_across_samples = mean(sparsity, na.rm = TRUE),
        sd_sparsity_across_samples = sd(sparsity, na.rm = TRUE),
        min_sparsity = min(sparsity, na.rm = TRUE),
        max_sparsity = max(sparsity, na.rm = TRUE),
        
        total_zero_entries = sum(zero_entries, na.rm = TRUE),
        total_entries = sum(total_entries, na.rm = TRUE),
        global_sparsity = total_zero_entries / total_entries,
        .groups = "drop"
    )

out_dir <- file.path(BASE_DIR, "HCA_sparsity_summary")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sample_out_csv <- file.path(out_dir, "HCA_sample_level_sparsity.csv")
summary_out_csv <- file.path(out_dir, "HCA_average_sparsity_across_samples.csv")

write_csv(sparsity_df, sample_out_csv)
write_csv(summary_df, summary_out_csv)

message("========================================")
message("Sample-level sparsity:")
print(sparsity_df)

message("========================================")
message("Average sparsity across samples:")
print(summary_df)

message("========================================")
message("Saved:")
message(sample_out_csv)
message(summary_out_csv)