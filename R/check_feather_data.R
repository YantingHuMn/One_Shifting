check_feather_data <- function(file_path) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(arrow)
    })

    cat("\n", strrep("=", 60), "\n")
    cat("File:", file_path, "\n")
    cat(strrep("=", 60), "\n")
    
    df <- read_feather(file_path)
    
    cat("Shape:", nrow(df), "rows x", ncol(df), "columns\n")
    cat("First column name:", colnames(df)[1], "\n")
    
    data_cols <- df[, -1, drop = FALSE]
    
    mat <- as.matrix(data_cols)
    
    # Check if all values are integers
    is_integer_vals <- all(mat == floor(mat), na.rm = TRUE)
    cat("\nAll values are integers:", is_integer_vals, "\n")
    print(mat[1:3,1:3])
    
    # Range
    min_val <- min(mat, na.rm = TRUE)
    max_val <- max(mat, na.rm = TRUE)
    cat("\nRange: [", min_val, ",", max_val, "]\n")
    
    # Sparsity (proportion of zeros)
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