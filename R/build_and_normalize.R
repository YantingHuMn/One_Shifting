# Generate count matrix: filter columns → remove 0 rows → cut columns → normalize → save
build_and_normalize <- function(filtered_bam_files, out_dir, bam_dir_name, norm_factors, keep_rows = NULL, keep_cols = NULL) {
    suppressPackageStartupMessages({
        library(arrow)
    })

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

    # remove 0 rows
    if (is.null(keep_rows)) {
        keep_rows <- rowSums(count_data[, -1] > 0, na.rm = TRUE) >= 2
    }
    count_data <- count_data[keep_rows, ]

    # cut columns
    if (!is.null(keep_cols)) {
        count_data <- count_data[, c(TRUE, keep_cols), drop = FALSE]
    }

    for (norm_val in norm_factors) {
        out_path <- file.path(save_dir, paste0("Count_Matrix_norm_by_", norm_val, ".feather"))

        pos_col <- count_data[, 1]
        count_matrix <- as.matrix(count_data[, -1])

        if (norm_val == "no_norm") {
            write_feather(count_data, out_path)

        } else if (norm_val == "standardize") {
            standardized_counts <- scale(count_matrix)
            standardized_counts[is.na(standardized_counts)] <- 0
            out_data <- cbind(pos_col, as.data.frame(standardized_counts))
            names(out_data)[1] <- names(count_data)[1]
            write_feather(out_data, out_path)

        } else {
            lib_sizes <- colSums(count_matrix)
            if (norm_val == "maximum") {
                current_norm_factor <- max(lib_sizes)
            } else {
                current_norm_factor <- as.numeric(norm_val)
            }
            normalized_counts <- sweep(count_matrix, 2, lib_sizes, "/") * current_norm_factor
            out_data <- cbind(pos_col, as.data.frame(normalized_counts))
            names(out_data)[1] <- names(count_data)[1]
            write_feather(out_data, out_path)
        }
    }

    return(list(keep_rows = keep_rows, keep_cols = keep_cols, count_data = count_data))
}
