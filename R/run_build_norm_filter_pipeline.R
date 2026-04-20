
source(file.path(dirname(sys.frame(1)$ofile), "count_matrix_function_with_qc.R"))
source(file.path(dirname(sys.frame(1)$ofile), "qc.R"))

source(file.path(dirname(sys.frame(1)$ofile), "cut_df_pct.R"))
source(file.path(dirname(sys.frame(1)$ofile), "get_filtered_samples.R"))
source(file.path(dirname(sys.frame(1)$ofile), "match_bam_files.R"))
source(file.path(dirname(sys.frame(1)$ofile), "build_and_normalize.R"))
source(file.path(dirname(sys.frame(1)$ofile), "normalize_and_save.R"))
source(file.path(dirname(sys.frame(1)$ofile), "apply_transformations.R"))


# Main Function: filter col → remove 0 row → cut col → normalize
#
# Two input modes:
#   file input (gene x cell): filter rows(gene) → cut rows(gene) → norm cols(cell) → transpose → save as cell x gene
#   dir  input (cell x gene): build from bam → filter rows(cell) → cut cols(gene) → norm cols(gene) → save as cell x gene
run_build_norm_filter_pipeline <- function(path1, path2, out_dir, norm_factor1, norm_factor2, filtered_percentile = 0.25,
                         zero_pct_max = NULL, pearson_min = NULL, spearman_min = NULL, lib_size_min = NULL,
                         histone_only = FALSE, transpose = FALSE) {
    suppressPackageStartupMessages({
        library(arrow)
    })

    is_file1 <- !dir.exists(path1) && file.exists(path1)
    is_file2 <- !dir.exists(path2) && file.exists(path2)

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

        cat("\n=== Processing V1 (initial, cell x gene) ===\n")
        v1_bam_files <- match_bam_files(path1, filtered_crf)
        v1_init <- build_and_normalize(v1_bam_files, out_dir, basename(path1), "no_norm", keep_rows = NULL, keep_cols = NULL)
        keep_rows <- v1_init$keep_rows
        v1_count_data <- v1_init$count_data
    }

    # Cut columns/rows depending on input mode
    # file input (gene x cell): cut_df_pct works on columns, so transpose first → cut genes as columns
    # dir  input (cell x gene): cut_df_pct works on columns → cut genes as columns (already correct)
    keep_cut <- NULL
    if (!is.null(zero_pct_max) || !is.null(pearson_min) || !is.null(spearman_min) || !is.null(lib_size_min)) {
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

    # V2 final
    cat("\n=== Processing V2 (final) ===\n")
    if (is_file2) {
        if (!exists("v2_count_data")) {
            v2_count_data <- read_feather(path2)
            v2_count_data <- v2_count_data[keep_rows, ]
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
    return(list(v1 = v1_result$count_data, v2 = v2_result$count_data))
}