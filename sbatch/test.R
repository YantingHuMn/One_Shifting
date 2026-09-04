suppressPackageStartupMessages({
    library(arrow)
    library(dplyr)
})


# ============================================================
# Path
# ============================================================

input_dir <- "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/Hrv39_INPUT_ATAC/Hrv39_ATAC"

output_summary <- file.path(
    input_dir,
    "input_before_after_log2_numeric_summary.csv"
)

output_gene_summary <- file.path(
    input_dir,
    "input_problem_gene_before_after_log2_summary.csv"
)


# ============================================================
# Find normalized input files
# ============================================================

input_files <- list.files(
    path = input_dir,
    pattern = "^Count_Matrix_norm_by_.*\\.feather$",
    full.names = TRUE
)

# Do not apply log2 transformation to standardized data,
# because standardized data may contain negative values
input_files <- input_files[
    !grepl(
        "norm_by_standardize\\.feather$",
        input_files
    )
]


if (length(input_files) == 0) {
    stop("No normalized input feather files found.")
}


cat("Found", length(input_files), "input files:\n")
cat(paste0("  ", basename(input_files)), sep = "\n")
cat("\n\n")


# Genes that showed extreme reconstruction values
genes_to_check <- c(
    "LINC00864",
    "FYN",
    "GSE1",
    "HOOK2",
    "DENND1A",
    "PTPRN2",
    "ZMIZ1",
    "MSI2",
    "CAMTA1",
    "RAD51B",
    "RAI1",
    "SEPT9",
    "NCOR2",
    "ZBTB7C"
)


file_summaries <- list()
gene_summaries <- list()


# ============================================================
# Check each file
# ============================================================

for (file_index in seq_along(input_files)) {

    file_path <- input_files[file_index]
    file_name <- basename(file_path)

    cat(
        "============================================================\n",
        "Checking file ", file_index, "/", length(input_files), ":\n",
        file_name, "\n",
        "============================================================\n",
        sep = ""
    )


    df <- read_feather(file_path)


    # First column is pos
    data_cols <- names(df)[-1]

    numeric_cols <- data_cols[
        vapply(
            df[data_cols],
            is.numeric,
            logical(1)
        )
    ]


    if (length(numeric_cols) == 0) {
        warning(
            "No numeric data columns found in: ",
            file_name
        )
        next
    }


    # Combine numeric input values
    raw_values <- unlist(
        df[numeric_cols],
        use.names = FALSE
    )


    # Basic checks before transformation
    n_na_raw <- sum(
        is.na(raw_values) &
        !is.nan(raw_values)
    )

    n_nan_raw <- sum(is.nan(raw_values))

    n_pos_inf_raw <- sum(
        raw_values == Inf,
        na.rm = TRUE
    )

    n_neg_inf_raw <- sum(
        raw_values == -Inf,
        na.rm = TRUE
    )

    n_negative_raw <- sum(
        raw_values < 0,
        na.rm = TRUE
    )


    finite_raw <- raw_values[
        is.finite(raw_values)
    ]


    # Apply log2(x + 1) only to valid nonnegative values
    log2_values <- rep(
        NA_real_,
        length(raw_values)
    )

    valid_for_log2 <- (
        is.finite(raw_values) &
        raw_values >= 0
    )

    log2_values[valid_for_log2] <-
        log2(
            raw_values[valid_for_log2] + 1
        )


    finite_log2 <- log2_values[
        is.finite(log2_values)
    ]


    file_summary <- data.frame(
        file = file_name,

        n_rows = nrow(df),

        n_columns_total = ncol(df),

        n_numeric_columns =
            length(numeric_cols),

        n_na_raw = n_na_raw,

        n_nan_raw = n_nan_raw,

        n_pos_inf_raw = n_pos_inf_raw,

        n_neg_inf_raw = n_neg_inf_raw,

        n_negative_raw = n_negative_raw,

        raw_min = if (length(finite_raw) > 0) {
            min(finite_raw)
        } else {
            NA_real_
        },

        raw_median = if (length(finite_raw) > 0) {
            median(finite_raw)
        } else {
            NA_real_
        },

        raw_q99 = if (length(finite_raw) > 0) {
            unname(
                quantile(
                    finite_raw,
                    probs = 0.99
                )
            )
        } else {
            NA_real_
        },

        raw_q999 = if (length(finite_raw) > 0) {
            unname(
                quantile(
                    finite_raw,
                    probs = 0.999
                )
            )
        } else {
            NA_real_
        },

        raw_max = if (length(finite_raw) > 0) {
            max(finite_raw)
        } else {
            NA_real_
        },

        log2_min = if (length(finite_log2) > 0) {
            min(finite_log2)
        } else {
            NA_real_
        },

        log2_median = if (length(finite_log2) > 0) {
            median(finite_log2)
        } else {
            NA_real_
        },

        log2_q99 = if (length(finite_log2) > 0) {
            unname(
                quantile(
                    finite_log2,
                    probs = 0.99
                )
            )
        } else {
            NA_real_
        },

        log2_q999 = if (length(finite_log2) > 0) {
            unname(
                quantile(
                    finite_log2,
                    probs = 0.999
                )
            )
        } else {
            NA_real_
        },

        log2_max = if (length(finite_log2) > 0) {
            max(finite_log2)
        } else {
            NA_real_
        },

        log2_values_above_20 = sum(
            log2_values > 20,
            na.rm = TRUE
        ),

        log2_values_above_100 = sum(
            log2_values > 100,
            na.rm = TRUE
        ),

        log2_values_above_1024 = sum(
            log2_values > 1024,
            na.rm = TRUE
        )
    )


    file_summaries[[file_name]] <- file_summary


    # ========================================================
    # Check previously problematic genes
    # ========================================================

    available_genes <- intersect(
        genes_to_check,
        names(df)
    )


    gene_summary <- bind_rows(
        lapply(available_genes, function(gene) {

            raw_x <- df[[gene]]

            log2_x <- rep(
                NA_real_,
                length(raw_x)
            )

            valid_x <- (
                is.finite(raw_x) &
                raw_x >= 0
            )

            log2_x[valid_x] <-
                log2(raw_x[valid_x] + 1)


            finite_raw_x <- raw_x[
                is.finite(raw_x)
            ]

            finite_log2_x <- log2_x[
                is.finite(log2_x)
            ]


            data.frame(
                file = file_name,
                gene = gene,

                n_total = length(raw_x),

                n_zero = sum(
                    raw_x == 0,
                    na.rm = TRUE
                ),

                n_nonzero = sum(
                    raw_x != 0,
                    na.rm = TRUE
                ),

                nonzero_percentage =
                    100 *
                    mean(
                        raw_x != 0,
                        na.rm = TRUE
                    ),

                raw_min = if (
                    length(finite_raw_x) > 0
                ) {
                    min(finite_raw_x)
                } else {
                    NA_real_
                },

                raw_median = if (
                    length(finite_raw_x) > 0
                ) {
                    median(finite_raw_x)
                } else {
                    NA_real_
                },

                raw_mean = if (
                    length(finite_raw_x) > 0
                ) {
                    mean(finite_raw_x)
                } else {
                    NA_real_
                },

                raw_sd = if (
                    length(finite_raw_x) > 1
                ) {
                    sd(finite_raw_x)
                } else {
                    NA_real_
                },

                raw_max = if (
                    length(finite_raw_x) > 0
                ) {
                    max(finite_raw_x)
                } else {
                    NA_real_
                },

                log2_min = if (
                    length(finite_log2_x) > 0
                ) {
                    min(finite_log2_x)
                } else {
                    NA_real_
                },

                log2_median = if (
                    length(finite_log2_x) > 0
                ) {
                    median(finite_log2_x)
                } else {
                    NA_real_
                },

                log2_mean = if (
                    length(finite_log2_x) > 0
                ) {
                    mean(finite_log2_x)
                } else {
                    NA_real_
                },

                log2_sd = if (
                    length(finite_log2_x) > 1
                ) {
                    sd(finite_log2_x)
                } else {
                    NA_real_
                },

                log2_max = if (
                    length(finite_log2_x) > 0
                ) {
                    max(finite_log2_x)
                } else {
                    NA_real_
                }
            )
        })
    )


    gene_summaries[[file_name]] <- gene_summary


    print(file_summary)

    cat("\nProblem gene summary:\n")

    print(
        gene_summary %>%
            arrange(desc(log2_max))
    )

    cat("\n")


    rm(
        df,
        raw_values,
        log2_values,
        finite_raw,
        finite_log2
    )

    gc()
}


# ============================================================
# Combine and save
# ============================================================

all_file_summary <- bind_rows(
    file_summaries
)

all_gene_summary <- bind_rows(
    gene_summaries
)


write.csv(
    all_file_summary,
    output_summary,
    row.names = FALSE
)

write.csv(
    all_gene_summary,
    output_gene_summary,
    row.names = FALSE
)


# ============================================================
# Final results
# ============================================================

cat("\n============================================================\n")
cat("INPUT SUMMARY BEFORE AND AFTER LOG2(x + 1)\n")
cat("============================================================\n\n")

print(
    all_file_summary %>%
        select(
            file,
            n_negative_raw,
            raw_median,
            raw_q99,
            raw_q999,
            raw_max,
            log2_median,
            log2_q99,
            log2_q999,
            log2_max,
            log2_values_above_20,
            log2_values_above_100,
            log2_values_above_1024
        )
)


cat("\n============================================================\n")
cat("LINC00864 SUMMARY\n")
cat("============================================================\n\n")

print(
    all_gene_summary %>%
        filter(gene == "LINC00864") %>%
        select(
            file,
            gene,
            n_zero,
            n_nonzero,
            nonzero_percentage,
            raw_min,
            raw_mean,
            raw_max,
            log2_min,
            log2_mean,
            log2_max
        )
)


cat("\nSaved overall input summary to:\n")
cat(output_summary, "\n")

cat("\nSaved problem gene summary to:\n")
cat(output_gene_summary, "\n")