check_feather_data <- function(input) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(arrow)
    })

    cat("\n", strrep("=", 60), "\n")

    if (is.character(input)) {
        cat("File:", input, "\n")
        df <- read_feather(input)
    } else {
        cat("Input: data.frame\n")
        df <- input
    }

    cat(strrep("=", 60), "\n")
    cat("Shape:", nrow(df), "rows x", ncol(df), "columns\n")
    cat("First column name:", colnames(df)[1], "\n")

    data_cols <- df[, -1, drop = FALSE]
    mat <- as.matrix(data_cols)

    is_integer_vals <- all(mat == floor(mat), na.rm = TRUE)
    cat("\nAll values are integers:", is_integer_vals, "\n")
    print(df[1:min(3, nrow(df)), 1:min(4, ncol(df))])

    min_val <- min(mat, na.rm = TRUE)
    max_val <- max(mat, na.rm = TRUE)
    cat("\nRange: [", min_val, ",", max_val, "]\n")

    total_elements <- length(mat)
    num_zeros <- sum(mat == 0, na.rm = TRUE)
    sparsity <- num_zeros / total_elements
    cat("\nSparsity (proportion of zeros):", round(sparsity * 100, 2), "%\n")

    return(invisible(list(
        is_integer = is_integer_vals,
        range = c(min_val, max_val),
        sparsity = sparsity
    )))
}