# Build Count Matrix and Normalize
# Post: Build a count matrix from BAM files using count_matrix_function_with_qc, apply row/column filtering, normalize by specified factors, optionally transpose, and save as Feather files.
# Parameter:
#   filtered_bam_files : Character vector of BAM file paths (already QC-filtered).
#   out_dir            : Base output directory.
#   bam_dir_name       : Subdirectory name created under out_dir for this dataset.
#   norm_factors       : Character vector of normalization methods. Supported: "no_norm", "standardize", "maximum", or a numeric string (e.g. "10000").
#   keep_rows          : Optional logical vector for row filtering. If NULL, rows with fewer than 2 nonzero values are removed. Default NULL.
#   keep_cols          : Optional logical vector for column filtering. Default NULL.
#   transpose          : Logical. If TRUE, transpose the result before saving. Default FALSE.
# Output: Returns (invisibly) a list with keep_rows, keep_cols, and count_data. Saves normalized Feather files to the subdirectory.
build_and_normalize <- function(filtered_bam_files, out_dir, bam_dir_name, norm_factors, keep_rows = NULL, keep_cols = NULL, transpose = FALSE) {
    suppressPackageStartupMessages({ library(arrow) })

    save_dir <- file.path(out_dir, bam_dir_name)
    dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

    count_matrix_function_with_qc(
        bam_path = filtered_bam_files,
        regions = 800,
        save_dir = save_dir,
        do_qc = FALSE,
        apply_transformation = FALSE
    )

    count_path <- file.path(save_dir, "Count_Matrix_orig.feather")
    count_data <- read_feather(count_path)

    if (is.null(keep_rows)) {
        keep_rows <- rowSums(count_data[, -1] > 0, na.rm = TRUE) >= 2
    }
    count_data <- count_data[keep_rows, ]

    if (!is.null(keep_cols)) {
        count_data <- count_data[, c(TRUE, keep_cols), drop = FALSE]
    }

    for (norm_val in norm_factors) {
        out_path <- file.path(save_dir, paste0("Count_Matrix_norm_by_", norm_val, ".feather"))

        pos_col <- count_data[[1]]
        count_matrix <- as.matrix(count_data[, -1])
        rownames(count_matrix) <- pos_col

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

    return(invisible(list(keep_rows = keep_rows, keep_cols = keep_cols, count_data = count_data)))
}