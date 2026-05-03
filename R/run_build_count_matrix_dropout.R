run_build_count_matrix_dropout <- function(path, out_dir, norm_factors) {
    suppressPackageStartupMessages({
        library(arrow)
    })

    save_dir <- out_dir
    dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

    count_data <- read_feather(path)
    for (i in 2:ncol(count_data)) {
        count_data[[i]] <- as.numeric(count_data[[i]])
    }

    pos_col <- count_data[, 1]
    count_matrix <- as.matrix(count_data[, -1])

    out_paths <- c()

    for (norm_val in norm_factors) {
        out_path <- file.path(save_dir,
                            paste0("Count_Matrix_norm_by_", norm_val, ".feather"))

        if (norm_val == "no_norm") {
            result <- count_matrix
        } else if (norm_val == "standardize") {
            result <- scale(count_matrix)
            result[is.na(result)] <- 0
        } else {
            lib_sizes <- colSums(count_matrix)
            if (norm_val == "maximum") {
                current_norm_factor <- max(lib_sizes)
            } else {
                current_norm_factor <- as.numeric(norm_val)
            }
            result <- sweep(count_matrix, 2, lib_sizes, "/") * current_norm_factor
        }

        result_df <- as.data.frame(result)
        result_df <- cbind(pos = pos_col, result_df)
        write_feather(result_df, out_path)

        out_paths <- c(out_paths, out_path)
    }

    invisible(out_paths)
}

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
out_dir <- args[2]
norm_factor_string <- args[3]
norm_factor <- unlist(strsplit(norm_factor_string, ","))

run_build_count_matrix_dropout(path = path, out_dir = out_dir, norm_factors = norm_factor)