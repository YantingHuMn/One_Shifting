# For file input (gene x cell): filter rows (genes) → normalize cols (cells) → transpose → save
# Normalize and Save (File Input)
# Post: For file-input mode (gene x cell): filter rows (genes), normalize columns (cells) by specified factors, optionally transpose to cell x gene, and save as Feather files.
# Parameter:
#   count_data    : Data frame with first column as row IDs and remaining columns as numeric counts (gene x cell).
#   out_dir       : Base output directory.
#   sub_dir_name  : Subdirectory name created under out_dir for this dataset.
#   norm_factors  : Character vector of normalization methods. Supported: "no_norm", "standardize", "maximum", or a numeric string.
#   keep_rows     : Optional logical vector for row (gene) filtering. Default NULL.
#   transpose     : Logical. If TRUE, transpose the result before saving. Default FALSE.
# Output: Returns (invisibly) a list with count_data. Saves normalized Feather files to the subdirectory.
normalize_and_save <- function(count_data, out_dir, sub_dir_name, norm_factors, keep_rows = NULL, transpose = FALSE) {
    # Load Libraries
    suppressPackageStartupMessages({ 
        library(arrow) 
    })

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