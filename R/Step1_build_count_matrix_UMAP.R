
run_build_count_matrix_UMAP <- function(path, out_dir, norm_factors, min_nonzero = 2) {
  
    # Load Libraries
    suppressPackageStartupMessages({
        library(arrow)
    })
    # create output directory
    save_dir <- file.path(out_dir, basename(dirname(path)))
    dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
    
    # read data
    count_data <- read_feather(path)
    for (i in 2:ncol(count_data)) {
        count_data[[i]] <- as.numeric(count_data[[i]])
    }
    
    # filter: keep rows with at least min_nonzero non-zero values
    keep_rows <- rowSums(count_data[, -1] > 0, na.rm = TRUE) >= min_nonzero
    count_data <- count_data[keep_rows, ]
    
    pos_col <- count_data[, 1]
    count_matrix <- as.matrix(count_data[, -1])
    rownames(count_matrix) <- pos_col
    
    out_paths <- c()
    
    # normalize and save
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
        
        result_t <- t(result)
        transposed_data <- as.data.frame(result_t)
        transposed_data$pos <- rownames(result_t)
        transposed_data <- transposed_data[, c("pos", setdiff(colnames(transposed_data), "pos"))]
        write_feather(transposed_data, out_path)
        
        out_paths <- c(out_paths, out_path)
    }
    
    invisible(out_paths)
}

args <- commandArgs(trailingOnly = TRUE)

path <- args[1]
out_dir <- args[2]
norm_factor_string <- args[3]
norm_factor <- unlist(strsplit(norm_factor_string, ","))

run_build_count_matrix_UMAP(path = path, out_dir = out_dir, norm_factor = norm_factor, norm_factor2 = norm_factor2, filtered_percentile = 0.25, lib_size_min = lib_size_min, transpose = transpose)
