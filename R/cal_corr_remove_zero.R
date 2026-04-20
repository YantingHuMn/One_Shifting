#  Helper: return matrices without any row/column filtering
cal_corr_remove_zero <- function(df1, df2, corr_type = "col") {
    X1 <- as.matrix(df1)
    X2 <- as.matrix(df2)

    if (!all(dim(X1) == dim(X2))) {
      print(paste("X1 dim:", paste(dim(X1), collapse = " ")))
      print(paste("X2 dim:", paste(dim(X2), collapse = " ")))
      X1[1:5, 1:5]
      X2[1:5, 1:5]
      stop("The dimensions of the two matrices do not match.")
    }

    stopifnot(all(dim(X1) == dim(X2)))

    # No filtering — return original matrices
    return(list(df1 = X1, df2 = X2))
}