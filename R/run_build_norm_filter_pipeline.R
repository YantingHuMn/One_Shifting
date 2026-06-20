
source("../One_Shifting/R/count_matrix_function_with_qc.R")
source("../One_Shifting/R/qc.R")

source("../One_Shifting/R/cut_df_pct.R")
source("../One_Shifting/R/get_filtered_samples.R")
source("../One_Shifting/R/match_bam_files.R")
source("../One_Shifting/R/build_and_normalize.R")
source("../One_Shifting/R/normalize_and_save.R")
source("../One_Shifting/R/apply_transformations.R")
source("../One_Shifting/R/dropout_multinomial.R")

# path1: INPUT
# path2: Ground Truth
# Build-Normalize-Filter Pipeline
# Post: End-to-end pipeline that reads two datasets (V1 and V2), applies QC filtering, builds count matrices, cuts low-quality features, normalizes, and saves final outputs. Supports both file input (gene x cell Feather) and directory input (BAM files producing cell x gene).
# Parameter:
#   path1               : Path to V1 data. Either a Feather file (gene x cell) or a directory of BAM files.
#   path2               : Path to V2 (ground truth) data. Either a Feather file or a directory of BAM files.
#   out_dir             : Base output directory for all results.
#   norm_factor1        : Character vector of normalization methods for V1.
#   norm_factor2        : Character vector of normalization methods for V2.
#   filtered_percentile : Numeric in (0, 1); QC percentile threshold for BAM filtering. Default 0.25.
#   zero_pct_max        : Maximum zero percentage for column cutting. Numeric, percentile string, or NULL. Default NULL.
#   pearson_min         : Minimum Pearson correlation for column cutting. Numeric, percentile string, or NULL. Default NULL.
#   spearman_min        : Minimum Spearman correlation for column cutting. Numeric, percentile string, or NULL. Default NULL.
#   lib_size_min        : Minimum library size for column cutting. Numeric, percentile string, or NULL. Default NULL.
#   histone_only        : Logical. If TRUE, keep only histone-marked CRF pairs. Default FALSE.
#   transpose           : Logical. If TRUE, transpose final output. Default FALSE.
# Output: A list with v1 and v2, each containing the final count data frame.
run_build_norm_filter_pipeline <- function(path1, path2 = NULL, out_dir, norm_factor1, norm_factor2 = NULL, filtered_percentile = 0.25,
                        zero_pct_max = NULL, pearson_min = NULL, spearman_min = NULL, lib_size_min = NULL,
                        histone_only = FALSE, transpose = FALSE, dropout_keep_par = NULL, dropout_target = c("none", "v1", "both")) {
                        
    # Load Libraries
    suppressPackageStartupMessages({
        library(arrow)
    })

    dropout_target <- match.arg(dropout_target)
    has_path2 <- !is.null(path2)

    is_file1 <- !dir.exists(path1) && file.exists(path1)
    is_file2 <- has_path2 && !dir.exists(path2) && file.exists(path2)

    if (is_file1) {
        # path1 is a file (gene x cell): read directly, skip bam building and QC
        # filter rows = filter genes (remove genes with < 2 nonzero cells)
        cat("\n=== Reading V1 from file (gene x cell) ===\n")
        v1_count_data <- read_feather(path1)
        keep_rows <- rowSums(v1_count_data[, -1] > 0, na.rm = TRUE) >= 2
        v1_count_data <- v1_count_data[keep_rows, ]
        v1_bam_files <- NULL
        cat(sprintf("  Kept %d / %d genes after row filter\n", sum(keep_rows), length(keep_rows)))
    } else {
        # path1 is a dir: QC + build → produces cell x gene
        filtered_crf <- get_filtered_samples(path1, filtered_percentile)

        if (histone_only) {
            histone_pattern <- "^H[0-9]"
            filtered_crf <- filtered_crf[sapply(filtered_crf, function(x) {
                parts <- strsplit(x, "-")[[1]]
                left <- parts[1]
                right <- paste(parts[-1], collapse = "-")
                grepl(histone_pattern, left) & grepl(histone_pattern, right)
            })]
            cat("After histone filter:", length(filtered_crf), "CRF pairs remaining\n")
        }

        cat("\n=== Processing V1 (initial, locus x CRF pairs) ===\n")
        v1_bam_files <- match_bam_files(path1, filtered_crf)
        v1_init <- build_and_normalize(v1_bam_files, out_dir, basename(path1), "no_norm", keep_rows = NULL, keep_cols = NULL)
        keep_rows <- v1_init$keep_rows
        v1_count_data <- v1_init$count_data
    }

    # Cut columns/rows depending on input mode
    # file input (gene x cell): cut_df_pct works on columns, so transpose first → cut genes as columns
    # dir  input (cell x gene): cut_df_pct works on columns → cut genes as columns (already correct)
    keep_cut <- NULL
    if (has_path2 && (!is.null(zero_pct_max) || !is.null(pearson_min) || !is.null(spearman_min) || !is.null(lib_size_min))) {
        if (is_file2) {
            cat("\n=== Reading V2 from file (for cut reference) ===\n")
            v2_count_data <- read_feather(path2)
            v2_count_data <- v2_count_data[keep_rows, ]
        } else {
            cat("\n=== Processing V2 (for cut reference) ===\n")
            if (!exists("filtered_crf")) {
                filtered_crf <- get_filtered_samples(path1, filtered_percentile)
            }
            v2_bam_files <- match_bam_files(path2, filtered_crf)
            v2_init <- build_and_normalize(v2_bam_files, out_dir, basename(path2), "no_norm",
                                           keep_rows = keep_rows, keep_cols = NULL)
            v2_count_data <- v2_init$count_data
        }

        v1_mat <- as.matrix(v1_count_data[, -1])
        v2_mat <- as.matrix(v2_count_data[, -1])

        if (is_file1) {
            # file input: gene x cell → transpose so cut_df_pct filters genes as columns
            cat("\n=== Cutting genes (transposing for cut_df_pct) ===\n")
            keep_cut <- cut_df_pct(t(v1_mat), t(v2_mat), zero_pct_max, pearson_min, spearman_min, lib_size_min)
        } else {
            # dir input: cell x gene → genes are already columns
            cat("\n=== Cutting columns (genes) ===\n")
            keep_cut <- cut_df_pct(v1_mat, v2_mat, zero_pct_max, pearson_min, spearman_min, lib_size_min)
        }
    }

    # V1 final
    if (!is.null(dropout_keep_par) && dropout_target %in% c("v1", "both")) {
        cat(sprintf("\n=== Applying multinomial dropout to V1: keep_par = %s ===\n", dropout_keep_par))

        v1_count_data <- dropout_multinomial(
            v1_count_data,
            keep_par = dropout_keep_par,
            cell_axis = if (is_file1) "col" else "row"
        )
    }

    cat("\n=== Processing V1 (final) ===\n")
    if (is_file1) {
        # file input: keep_cut filters rows (genes), then norm by col (cell), then transpose
        v1_result <- normalize_and_save(v1_count_data, out_dir, basename(dirname(path1)), norm_factor1,
                                        keep_rows = keep_cut, transpose = transpose)
    } else {
        # dir input: keep_cut filters cols (genes)
        v1_result <- build_and_normalize(v1_bam_files, out_dir, basename(path1), norm_factor1,
                                         keep_rows = keep_rows, keep_cols = keep_cut, transpose = transpose)
    }

    if (has_path2) {
        # V2 final
        cat("\n=== Processing V2 (final) ===\n")
        if (is_file2) {
            if (!exists("v2_count_data")) {
                v2_count_data <- read_feather(path2)
                v2_count_data <- v2_count_data[keep_rows, ]
            }

            if (!is.null(dropout_keep_par) && dropout_target == "both") {
                cat(sprintf("\n=== Applying multinomial dropout to V2: keep_par = %s ===\n", dropout_keep_par))

                v2_count_data <- dropout_multinomial(
                    v2_count_data,
                    keep_par = dropout_keep_par,
                    cell_axis = "col"
                )
            }

            v2_result <- normalize_and_save(v2_count_data, out_dir, basename(dirname(path2)), norm_factor2,
                                            keep_rows = keep_cut, transpose = transpose)
        } else {
            if (!exists("v2_bam_files") || is.null(v2_bam_files)) {
                if (!exists("filtered_crf")) {
                    filtered_crf <- get_filtered_samples(path1, filtered_percentile)
                }
                v2_bam_files <- match_bam_files(path2, filtered_crf)
            }
            v2_result <- build_and_normalize(v2_bam_files, out_dir, basename(path2), norm_factor2,
                                             keep_rows = keep_rows, keep_cols = keep_cut, transpose = transpose)
        }

        cat("\n=== Pipeline complete ===\n")
        return(invisible(list(v1 = v1_result$count_data, v2 = v2_result$count_data)))
    } else {
        cat("\n=== Pipeline complete ===\n")
        return(invisible(list(v1 = v1_result$count_data)))
    }
}