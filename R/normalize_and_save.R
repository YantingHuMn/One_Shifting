# For file input (gene x cell): filter rows (genes) → normalize cols (cells) → transpose → save
normalize_and_save <- function(count_data, out_dir, sub_dir_name, norm_factors, keep_rows = NULL, transpose = FALSE) {
    suppressPackageStartupMessages({ library(arrow) })

    save_dir <- file.path(out_dir, sub_dir_name)
    dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

    # filter rows (genes)
    if (!is.null(keep_rows)) {
        count_data <- count_data[keep_rows, , drop = FALSE]
    }
    
    for (norm_val in norm_factors) {
        out_path <- file.path(save_dir, paste0("Count_Matrix_norm_by_", norm_val, ".feather"))

        id_col <- count_data[[1]]
        count_matrix <- as.matrix(count_data[, -1])
        rownames(count_matrix) <- id_col

        if (norm_val == "no_norm") {
            result_matrix <- count_matrix

        } else if (norm_val == "standardize") {
            result_matrix <- scale(count_matrix)
            result_matrix[is.na(result_matrix)] <- 0

        } else {
            lib_sizes <- colSums(count_matrix)
            if (norm_val == "maximum") {
                current_norm_factor <- max(lib_sizes)
            } else {
                current_norm_factor <- as.numeric(norm_val)
            }
            result_matrix <- sweep(count_matrix, 2, lib_sizes, "/") * current_norm_factor
        }

        if (transpose) {
            result_matrix <- t(result_matrix)
        }

        out_data <- as.data.frame(result_matrix)
        out_data$pos <- rownames(result_matrix)
        out_data <- out_data[, c("pos", setdiff(colnames(out_data), "pos"))]
        write_feather(out_data, out_path)
    }

    return(invisible(list(count_data = count_data)))
}