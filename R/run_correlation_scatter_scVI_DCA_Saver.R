args <- commandArgs(trailingOnly = TRUE)

path1 <- args[1] # ground truth
path2 <- args[2] # rep1
path3 <- args[3] # vae/dca/scvi/etc reconstruction
out_dir <- args[4]
save_csv_dir <- args[5]
corr <- args[6]  # "col" or "row"
y_title <- if (length(args) >= 7) args[7] else NULL
x_title <- if (length(args) >= 8) args[8] else NULL
# corr_method <- if (length(args) >= 9) args[9] else "pearson"  # "pearson" or "spearman"

source("../One_Shifting/R/row_cor_generic.R")
source("../One_Shifting/R/col_cor_generic.R")
source("../One_Shifting/R/cal_corr_remove_zero.R")


#  Main function: correlation scatter plot between reconstruction and original
run_correlation_scatter <- function(path1, path2, path3, out_dir,
                                    factor, V2_norm_factor, this_trans_factor,
                                    V2_trans_factor, save_csv_dir, corr,
                                    y_title = NULL, x_title = NULL,
                                    corr_method = "pearson",
                                    filter_zero_gt = FALSE) {

    suppressPackageStartupMessages({
        library(jsonlite)
        library(arrow)
        library(dplyr)
        library(ggplot2)
    })

    if (!dir.exists(save_csv_dir)) {
        dir.create(save_csv_dir, recursive = TRUE)
    }

    if (!corr %in% c("col", "row")) {
        stop("Error: corr parameter must be either 'col' or 'row'")
    }
    if (!corr_method %in% c("pearson", "spearman")) {
        stop("Error: corr_method must be either 'pearson' or 'spearman'")
    }

    filter_tag <- if (filter_zero_gt) "filtered" else "unfiltered"
    cat(paste("Calculating", corr_method, "correlation between:", corr,
              "| filter_zero_gt:", filter_zero_gt, "\n"))

    # Create Label
    label3 <- "Input"
    label4 <- "Output"

    # Override axis labels if provided
    if (!is.null(x_title)) label3 <- paste0(label4, "_", x_title)  # x-axis (path3) reconstructed/output
    if (!is.null(y_title)) label4 <- paste0(label3,"_", y_title)  # y-axis (path2) input

    subtitle_text <- ""

    #  Load data
    df1 <- read_feather(path1)  # ground truth
    df2 <- read_feather(path2)  # rep1 (control)
    df3 <- read_feather(path3)  # reconstruction (treatment)

    col1 <- colnames(df1)
    col2 <- colnames(df2)
    all_equal <- identical(col1, col2)
    print(paste("colnames for df1 and df2 are all equal:", all_equal))

    # Drop pos column if present
    if ("pos" %in% names(df1)) df1 <- df1[, !(names(df1) %in% "pos")]
    if ("pos" %in% names(df2)) df2 <- df2[, !(names(df2) %in% "pos")]
    if ("pos" %in% names(df3)) df3 <- df3 %>% dplyr::select(-pos)

    # Filter out all-zero columns/rows in ground truth (df1), sync df2 & df3
    n_before <- if (corr == "col") ncol(df1) else nrow(df1)

    if (filter_zero_gt) {
        if (corr == "col") {
            # Remove columns where ground truth is all zero
            non_zero_cols <- colSums(df1 != 0, na.rm = TRUE) > 0
            df1 <- df1[, non_zero_cols, drop = FALSE]
            df2 <- df2[, non_zero_cols, drop = FALSE]
            df3 <- df3[, non_zero_cols, drop = FALSE]
        } else {
            # Remove rows where ground truth is all zero
            non_zero_rows <- rowSums(df1 != 0, na.rm = TRUE) > 0
            df1 <- df1[non_zero_rows, , drop = FALSE]
            df2 <- df2[non_zero_rows, , drop = FALSE]
            df3 <- df3[non_zero_rows, , drop = FALSE]
        }
    }

    n_after <- if (corr == "col") ncol(df1) else nrow(df1)
    n_removed <- n_before - n_after
    cat(paste("filter_zero_gt:", filter_zero_gt,
              "| Before:", n_before, "| After:", n_after,
              "| Removed:", n_removed, "\n"))

    # Count columns with non-zero values in reconstruction
    not_zero_counts <- colSums(df3 != 0, na.rm = TRUE)
    non_zero_column_count <- sum(not_zero_counts > 0)
    non_zero_column_count

    #  Compute correlations: reconstruction vs ground truth
    print("Dimension after remove 0")
    results <- cal_corr_remove_zero(df1, df3, corr_type = corr)
    remove_zero_ground_truth <- results$df1
    dim(remove_zero_ground_truth)
    remove_zero_vae_10p <- results$df2
    dim(remove_zero_vae_10p)

    if (corr == "col") {
        r_vae_10p <- col_cor_generic(remove_zero_ground_truth, remove_zero_vae_10p, method = corr_method)
    } else {
        r_vae_10p <- row_cor_generic(remove_zero_ground_truth, remove_zero_vae_10p, method = corr_method)
    }

    head(r_vae_10p)
    non_zero_count <- sum(r_vae_10p != 0, na.rm = TRUE)
    non_zero_count

    #  Compute correlations: original input rep1 vs ground truth
    results <- cal_corr_remove_zero(df1, df2, corr_type = corr)
    remove_zero_ground_truth <- results$df1
    remove_zero_orig_10p <- results$df2

    if (corr == "col") {
        r_orig_10p <- col_cor_generic(remove_zero_ground_truth, remove_zero_orig_10p, method = corr_method)
    } else {
        r_orig_10p <- row_cor_generic(remove_zero_ground_truth, remove_zero_orig_10p, method = corr_method)
    }

    non_zero_count <- sum(r_orig_10p != 0, na.rm = TRUE)
    non_zero_count

    #  Prepare data for scatter plot
    # Replace NA with 0 before building data.frame
    r_vae_10p[is.na(r_vae_10p)]   <- 0
    r_orig_10p[is.na(r_orig_10p)] <- 0

    df_vae <- data.frame(name = names(r_vae_10p),  value = as.numeric(r_vae_10p),  stringsAsFactors = FALSE)
    df_orig <- data.frame(name = names(r_orig_10p), value = as.numeric(r_orig_10p), stringsAsFactors = FALSE)

    df_merged <- full_join(df_vae, df_orig, by = "name", suffix = c("_vae", "_orig"))

    df_merged <- df_merged %>%
        mutate(
            VAE  = coalesce(value_vae, 0),
            ORIG = coalesce(value_orig, 0)
        ) %>%
        select(Value = name, VAE, ORIG)

    intersection_df <- df_merged

    #  Points to highlight on the scatter plot
    highlight_points <- c("H3K4me3-H3K4me3", "H3K4me3-H3K9ac", "H3K4me3-POLR2AphosphoS2",
                          "H3K27me3-H3K27me3", "CDK8-H3K4me3", "H3K27me3-MLL4_MLL2_KMT2B",
                          "CBP_CREBBP-H3K4me3", "H3K9ac-H3K9ac", "H3K27ac-H3K4me3",
                          "H3K27ac-H3K27ac")

    n_points <- nrow(intersection_df)

    #  Compute summary statistics for subtitle
    x_avg <- mean(intersection_df$VAE)

    intersection_df$residual <- intersection_df$ORIG - intersection_df$VAE
    residual_sum <- sum(intersection_df$residual)
    residual_avg <- mean(intersection_df$residual)

    subtitle_text_with_stats <- paste0(
        subtitle_text,
        "(n = ", n_points, " pairs)",
        " | Corr_type: ", corr,
        " | x-axis_mean: ", round(x_avg, 4),
        "\n| Residual_sum: ", round(residual_sum, 4),
        " | Residual_mean: ", round(residual_avg, 4)
    )

    # Add filter info to subtitle
    if (filter_zero_gt) {
        subtitle_text_with_stats <- paste0(
            subtitle_text_with_stats,
            "\n| GT_zero_filtered: removed ", n_removed, " all-zero ", corr, "s (", n_before, " -> ", n_after, ")"
        )
    }

    # Count points above, below, and on the diagonal
    intersection_df$position <- case_when(
        intersection_df$ORIG > intersection_df$VAE ~ "above",
        intersection_df$ORIG < intersection_df$VAE ~ "below",
        TRUE ~ "on"
    )
    n_above <- sum(intersection_df$position == "above")
    n_below <- sum(intersection_df$position == "below")
    n_on    <- sum(intersection_df$position == "on")

    subtitle_text_with_stats <- paste0(
        subtitle_text_with_stats,
        "\nAbove: ", n_above, ", Below: ", n_below, ", On line: ", n_on
    )

    #  Method-specific plot styling
    if (corr_method == "pearson") {
        point_color <- "blue"
        point_alpha <- 0.7
        file_prefix <- "d_pearson_"
        method_label <- "Pearson"
    } else {
        point_color <- "darkred"
        point_alpha <- 0.7
        file_prefix <- "g_spearman_"
        method_label <- "Spearman"
    }

    # Add filter tag to file prefix
    if (filter_zero_gt) {
        file_prefix <- paste0(file_prefix, "noGTzero_")
    }

    #  Generate scatter plot
    p <- ggplot(intersection_df, aes(x = VAE, y = ORIG)) +
        geom_point(color = point_color, size = 2, alpha = point_alpha) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
        geom_smooth(method = "lm", se = FALSE, color = "blue", alpha = 0.2, linewidth = 0.5) +
        geom_text(data = intersection_df %>% filter(Value %in% highlight_points),
                  aes(label = Value),
                  hjust = 0.5, vjust = -0.5, size = 3, color = "red") +
        labs(title = paste0(method_label, " (", corr, ") V1: ", this_trans_factor, ",", factor, " V2: ", V2_trans_factor, ",", V2_norm_factor,
                            if (filter_zero_gt) " [GT-zero filtered]" else ""),
             subtitle = subtitle_text_with_stats,
             x = label3, y = label4) +
        scale_x_continuous(limits = c(-0.05, 1)) +
        scale_y_continuous(limits = c(-0.05, 1)) +
        theme_bw() +
        theme(plot.subtitle = element_text(size = 9, color = "gray30"))

    #  Save plot, scatter CSV, and summary CSV
    plot_filename <- paste0(file_prefix, corr, "_scatter_v1_trans_", this_trans_factor,
                            "_norm_", factor, "_v2_trans_", V2_trans_factor,
                            "_norm_", V2_norm_factor, ".png")
    ggsave(file.path(out_dir, plot_filename), p, width = 6.5, height = 4.5, dpi = 300)
    print(paste0(method_label, " scatter plot saved to ", file.path(out_dir, plot_filename)))

    csv_filename <- paste0(file_prefix, corr, "_scatter_v1_trans_", this_trans_factor,
                           "_norm_", factor, "_v2_trans_", V2_trans_factor,
                           "_norm_", V2_norm_factor, ".csv")
    write.csv(intersection_df, file.path(out_dir, csv_filename), row.names = FALSE)

    # Append summary row to cumulative CSV
    summary_row <- data.frame(
        REP1_trans = this_trans_factor,
        REP1_norm = factor,
        corr_type = corr,
        x_mean = round(x_avg, 4),
        residual_mean = round(residual_avg, 4),
        filter_zero_gt = filter_zero_gt,
        n_removed = n_removed,
        stringsAsFactors = FALSE
    )

    if (corr == "col") {
        obj <- "gene"
    } else if (corr == "row") {
        obj <- "cell"
    } else {
        stop("Error: `corr` direction must be 'col' or 'row'")
    }

    save_csv_path <- paste0(save_csv_dir, "/plots_summary_", corr_method, "_", corr, "_", obj,
                            if (filter_zero_gt) "_noGTzero" else "",
                            ".csv")

    if (file.exists(save_csv_path)) {
        existing_data <- read.csv(save_csv_path, stringsAsFactors = FALSE)
        updated_data  <- rbind(existing_data, summary_row)
        write.csv(updated_data, save_csv_path, row.names = FALSE)
    } else {
        write.csv(summary_row, save_csv_path, row.names = FALSE)
    }

    print(paste0("Summary statistics saved to ", save_csv_path))
}


# Run all 4 combinations: {pearson, spearman} x {unfiltered, filtered} 

# # 1. Pearson, unfiltered (original)
# run_correlation_scatter(
#     path1 = path1, path2 = path2, path3 = path3,
#     out_dir = out_dir, saved_models_dir = saved_models_dir,
#     factor = factor, V2_norm_factor = V2_norm_factor,
#     this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
#     save_csv_dir = save_csv_dir, corr = corr,
#     y_title = y_title, x_title = x_title,
#     corr_method = "pearson", filter_zero_gt = FALSE
# )

# # 2. Pearson, GT-zero filtered
# run_correlation_scatter(
#     path1 = path1, path2 = path2, path3 = path3,
#     out_dir = out_dir, saved_models_dir = saved_models_dir,
#     factor = factor, V2_norm_factor = V2_norm_factor,
#     this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
#     save_csv_dir = save_csv_dir, corr = corr,
#     y_title = y_title, x_title = x_title,
#     corr_method = "pearson", filter_zero_gt = TRUE
# )

# # 3. Spearman, unfiltered (original)
# run_correlation_scatter(
#     path1 = path1, path2 = path2, path3 = path3,
#     out_dir = out_dir, saved_models_dir = saved_models_dir,
#     factor = factor, V2_norm_factor = V2_norm_factor,
#     this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
#     save_csv_dir = save_csv_dir, corr = corr,
#     y_title = y_title, x_title = x_title,
#     corr_method = "spearman", filter_zero_gt = FALSE
# )

# # 4. Spearman, GT-zero filtered
# run_correlation_scatter(
#     path1 = path1, path2 = path2, path3 = path3,
#     out_dir = out_dir, saved_models_dir = saved_models_dir,
#     factor = factor, V2_norm_factor = V2_norm_factor,
#     this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
#     save_csv_dir = save_csv_dir, corr = corr,
#     y_title = y_title, x_title = x_title,
#     corr_method = "spearman", filter_zero_gt = TRUE
# )

factor <- "no_norm"
V2_norm_factor <- "no_norm"
this_trans_factor <- "no_trans"
V2_trans_factor <- "no_trans"

# 1. Pearson, unfiltered
tryCatch(
    run_correlation_scatter(
        path1 = path1, path2 = path2, path3 = path3,
        out_dir = out_dir,
        factor = factor, V2_norm_factor = V2_norm_factor,
        this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
        save_csv_dir = save_csv_dir, corr = corr,
        y_title = y_title, x_title = x_title,
        corr_method = "pearson", filter_zero_gt = FALSE
    ),
    error = function(e) cat("ERROR in Pearson unfiltered:", conditionMessage(e), "\n")
)

# 2. Pearson, GT-zero filtered
tryCatch(
    run_correlation_scatter(
        path1 = path1, path2 = path2, path3 = path3,
        out_dir = out_dir,
        factor = factor, V2_norm_factor = V2_norm_factor,
        this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
        save_csv_dir = save_csv_dir, corr = corr,
        y_title = y_title, x_title = x_title,
        corr_method = "pearson", filter_zero_gt = TRUE
    ),
    error = function(e) cat("ERROR in Pearson filtered:", conditionMessage(e), "\n")
)

# 3. Spearman, unfiltered
tryCatch(
    run_correlation_scatter(
        path1 = path1, path2 = path2, path3 = path3,
        out_dir = out_dir,
        factor = factor, V2_norm_factor = V2_norm_factor,
        this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
        save_csv_dir = save_csv_dir, corr = corr,
        y_title = y_title, x_title = x_title,
        corr_method = "spearman", filter_zero_gt = FALSE
    ),
    error = function(e) cat("ERROR in Spearman unfiltered:", conditionMessage(e), "\n")
)

# 4. Spearman, GT-zero filtered
tryCatch(
    run_correlation_scatter(
        path1 = path1, path2 = path2, path3 = path3,
        out_dir = out_dir,
        factor = factor, V2_norm_factor = V2_norm_factor,
        this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
        save_csv_dir = save_csv_dir, corr = corr,
        y_title = y_title, x_title = x_title,
        corr_method = "spearman", filter_zero_gt = TRUE
    ),
    error = function(e) cat("ERROR in Spearman filtered:", conditionMessage(e), "\n")
)