# Transformation Function
# Post: Transformation: accept transformation -- "remove0", "libnorm", "log2p1", "minmaxnorm", "sqrt", "sqrt_minmaxnorm", "qnorm".
# Parameter: df: Data frame used for transformation
#            libnorm_type1: Type of library normalization ("libnorm", "libnorm-mean", "libnorm-median")
#            transformations: Vector of transformation steps to apply, default to ("remove0", "libnorm", "log2p1", "qnorm")
#            save_each_step: Boolean flag to save intermediate results after each transformation
#            save_dir: Folder path for saving output files
#            datasetName_full: Output file name prefix
# Output: a step/overall transformed df and save.
apply_transformations <- function(df, save_dir, datasetName_full, libnorm_type1 = "libnorm", transformations = NULL, save_each_step = TRUE) {
    suppressPackageStartupMessages({
        library(arrow)
        library(tibble)
        library(matrixStats)
        library(data.table)
        library(rtracklayer)
        library(preprocessCore)
    })

    result <- df
    applied <- c()
    pos_colname = "pos"

    if (pos_colname %in% colnames(result)) {
        rownames(result) <- result[[pos_colname]]
        result <- result[, !colnames(result) %in% pos_colname]  
    }

    if (is.null(transformations)) {
        transformations <- c("remove0", "libnorm", "log2p1", "qnorm")
    }

    if (libnorm_type1 == "libnorm-mean") {
        norm_scale_factor = 51633
    } else if (libnorm_type1 == "libnorm-median") {
        norm_scale_factor = 5704
    } else if (libnorm_type1 == "libnorm") {
        norm_scale_factor = 1E6
    }

    for (t in transformations) {
        if (t == "remove0") {
            result <- result[rowSums(result) != 0, ] # remove all 0 row
            if (nrow(result) == 0) {
                stop("Error: all rows have zero counts; nothing left after filtering.")
            }
        } else if (t == "libnorm") {
            pol2_df <- result
            col_names <- colnames(result)
            zero_cols_idx <- which(colSums(pol2_df == 0) == nrow(pol2_df)) # if the whole col = 0
            if (length(zero_cols_idx) > 0) {
                zero_cols_names <- colnames(pol2_df)[zero_cols_idx]
                pol2_df_filtered <- pol2_df[, -zero_cols_idx]
                print(paste("Remove", length(zero_cols_idx), "column with all 0"))
                print(paste("Rmoved columns' name:", paste(zero_cols_names, collapse = ", ")))
                result <- pol2_df_filtered
            }
            result <- sweep(result, 2, colSums(result), FUN = "/") * norm_scale_factor

            if (length(zero_cols_idx) > 0) {
                result_with_zeros <- data.frame(matrix(0, nrow = nrow(result), ncol = length(col_names)))
                colnames(result_with_zeros) <- col_names
                rownames(result_with_zeros) <- rownames(result)
                current_col_names <- colnames(result)
                result_with_zeros[, current_col_names] <- result[, current_col_names]
                print(paste("Added back", length(zero_cols_idx), "columns with all 0"))
                result <- result_with_zeros
            }
        } else if (t == "log2p1") {
            result <- log2(result + 1)
        } else if (t == "minmaxnorm") {
            result <- sweep(result, 2, apply(result, 2, min), FUN = "-")
            result <- sweep(result, 2, apply(result, 2, function(x) max(x) - min(x)), FUN = "/")
        } else if (t == "sqrt") {
            result <- sqrt(result)
        } else if (t == "sqrt_minmaxnorm") {
            result <- sqrt(result)
            result <- sweep(result, 2, apply(result, 2, min), FUN = "-")
            result <- sweep(result, 2, apply(result, 2, function(x) max(x) - min(x)), FUN = "/")
        } else if (t == "qnorm") {
            old_rownames <- rownames(result)
            old_colnames <- colnames(result)  
            result <- normalize.quantiles(as.matrix(result), copy = TRUE)
            rownames(result) <- old_rownames 
            colnames(result) <- old_colnames
        } else {
            warning(paste0("Unrecognized transformation: ", t))
            next
        }

        applied <- c(applied, t)

        if (save_each_step == TRUE) {
            result_to_save <- rownames_to_column(as.data.frame(result), var = pos_colname)
            file_name <- paste0(datasetName_full, "_", libnorm_type1, "_", paste(applied, collapse = "_"), ".feather")
            save_path <- file.path(save_dir, file_name)
            write_feather(result_to_save, save_path)
        }
    }

    result_to_save <- rownames_to_column(as.data.frame(result), var = pos_colname)
    file_name <- paste0(datasetName_full, "_", libnorm_type1, "_all_transformed.feather")
    save_path <- file.path(save_dir, file_name)
    write_feather(result_to_save, save_path)

    return(result)
}