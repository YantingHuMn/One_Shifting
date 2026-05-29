args <- commandArgs(trailingOnly = TRUE)

path1 <- args[1] # ground truth
path2 <- args[2] # rep1
path3 <- args[3] # vae/dca/scvi/etc reconstruction
out_dir <- args[4]
saved_models_dir <- args[5]
factor <- args[6]
V2_norm_factor <- args[7]
this_trans_factor <- args[8]
V2_trans_factor <- args[9]
save_csv_dir <- args[10]
corr <- args[11]  # "col" or "row"
y_title <- if (length(args) >= 12) args[12] else NULL # V1
x_title <- if (length(args) >= 13) args[13] else NULL # method
# corr_method <- if (length(args) >= 14) args[14] else "pearson"  # "pearson" or "spearman"
data_mode <- args[14]

source("../One_Shifting/R/row_cor_generic.R")
source("../One_Shifting/R/col_cor_generic.R")
source("../One_Shifting/R/cal_corr_remove_zero.R")


#  Main function: correlation scatter plot between reconstruction and original
run_correlation_scatter <- function(path1, path2, path3, out_dir, saved_models_dir,
                                    factor, V2_norm_factor, this_trans_factor,
                                    V2_trans_factor, save_csv_dir, corr,
                                    y_title = NULL, x_title = NULL,
                                    corr_method = "pearson",
                                    filter_zero_gt = FALSE,
                                    data_mode = "default") {

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
    if (!data_mode %in% c("default", "v1_trans_v2_trans", "v1_reverse", "v1_trans_v2_trans_norm_100000")) {
        stop("Error: data_mode must be 'default', 'v1_trans_v2_trans', 'v1_reverse', or 'v1_trans_v2_trans_norm_100000'")
    }

    # Helper: apply transformation
    apply_trans <- function(mat, trans) {
        if (trans == "no_trans") {
            return(mat)
        } else if (trans == "sqrt") {
            return(sqrt(mat))
        } else if (trans == "sqrt+1") {
            return(sqrt(mat + 1))
        } else if (trans == "log2") {
            return(log2(mat + 1))
        } else if (trans == "count+1") {
            return(mat + 1)
        } else if (trans == "log2(count+2)") {
            return(log2(mat + 2))
        } else {
            stop(paste("Unknown trans:", trans))
        }
    }

    # Helper: reverse transformation
    reverse_trans <- function(mat, trans) {
        if (trans == "no_trans") {
            return(mat)
        } else if (trans == "sqrt") {
            return(mat^2)
        } else if (trans == "sqrt+1") {
            return(mat^2 - 1)
        } else if (trans == "log2") {
            return(2^mat - 1)
        } else if (trans == "count+1") {
            return(mat - 1)
        } else if (trans == "log2(count+2)") {
            return(2^mat - 2)
        } else {
            stop(paste("Unknown trans:", trans))
        }
    }

    filter_tag <- if (filter_zero_gt) "filtered" else "unfiltered"
    cat(paste("Calculating", corr_method, "correlation between:", corr,
              "| filter_zero_gt:", filter_zero_gt,
              "| data_mode:", data_mode, "\n"))

    # Create Label
    label3 <- "Input"
    label4 <- "Output"

    # Override axis labels if provided
    if (!is.null(x_title)) label4 <- paste0(label4, "_", x_title)  # x-axis (path3) reconstructed/output
    if (!is.null(y_title)) label3 <- paste0(label3,"_", y_title)  # y-axis (path2) input

    #  Read best fold config from saved_models_dir
    fold_dirs <- list.dirs(saved_models_dir, recursive = FALSE, full.names = TRUE)
    fold_dirs <- fold_dirs[grepl("fold_", fold_dirs)]

    if (length(fold_dirs) > 0) {
        best_fold   <- NULL
        best_metric <- Inf
        best_config <- NULL

        for (fold_dir in fold_dirs) {
            config_file <- list.files(fold_dir, pattern = "_config\\.json$", full.names = TRUE)[1]
            if (is.na(config_file) || !file.exists(config_file)) next

            # Read as text and replace NaN with null before parsing
            json_text <- readLines(config_file, warn = FALSE)
            json_text <- gsub(' NaN', ' null', json_text)
            config <- fromJSON(paste(json_text, collapse = ""))

            # Check if outer_test_metric is valid; fall back to outer_test_loss
            metric_value <- config$outer_test_metric
            if (is.null(metric_value) || is.na(metric_value) || !is.finite(metric_value)) {
                metric_value <- config$outer_test_loss
            }

            # Only consider this fold if metric_value is a valid number
            if (!is.null(metric_value) && !is.na(metric_value) && is.finite(metric_value)) {
                if (metric_value < best_metric) {
                    best_metric <- metric_value
                    best_fold   <- fold_dir
                    best_config <- config
                }
            }
        }

        if (is.null(best_fold)) {
            stop("Error: No fold with valid (non-NaN) metrics found!")
        }

        #  Build subtitle from best fold config
        fold_name  <- basename(best_fold)
        model_type <- ifelse(is.null(best_config$model_type), "VAE", best_config$model_type)

        # Compatible with different config formats (trans1 vs trans, scVI vs DCA/VAE)
        trans_label <- if (!is.null(best_config$trans1)) best_config$trans1
                       else if (!is.null(best_config$trans)) best_config$trans
                       else "none"

        beta_val   <- if (is.null(best_config$beta)) 0 else best_config$beta
        nonzero_w  <- if (is.null(best_config$nonzero_weight)) "N/A" else best_config$nonzero_weight

        # Detect architecture format: scVI (n_hidden/n_latent/n_layers) vs DCA/VAE (hidden_dim1/hidden_dim2/latent_dim)
        if (!is.null(best_config$n_hidden)) {
            arch_str <- sprintf("n_hidden=%d, n_latent=%d, n_layers=%d",
                                best_config$n_hidden, best_config$n_latent, best_config$n_layers)
        } else {
            arch_str <- sprintf("%d-%d-%d",
                                best_config$hidden_dim1, best_config$hidden_dim2, best_config$latent_dim)
        }

        subtitle_text <- sprintf(
            "%s | %s | Input: %d | Threshold: %.3f | Trans: %s | Arch: %s |\nBeta: %.2f | Nonzero_weight: %s",
            fold_name, model_type, best_config$input_dim, best_config$threshold,
            trans_label, arch_str, beta_val, as.character(nonzero_w)
        )
    } else {
        # MLP mode: no subtitle
        subtitle_text <- ""
    }

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

    # Apply data_mode transformations
    if (data_mode == "v1_trans_v2_trans") {
        mat1 <- as.matrix(df1)
        mat1 <- apply_trans(mat1, this_trans_factor)
        df1 <- as.data.frame(mat1)

        cat(paste("data_mode: v1_trans_v2_trans | ground truth trans:", this_trans_factor, "\n"))
    } else if (data_mode == "v1_reverse") {
        mat2 <- as.matrix(df2)
        mat2 <- reverse_trans(mat2, this_trans_factor)
        df2 <- as.data.frame(mat2)

        mat3 <- as.matrix(df3)
        mat3 <- reverse_trans(mat3, this_trans_factor)
        df3 <- as.data.frame(mat3)

        cat(paste("data_mode: v1_reverse | reversed input & reconstruction trans:", this_trans_factor, "\n"))
    } else if (data_mode == "v1_trans_v2_trans_norm_100000") {
        # Same as v1_trans_v2_trans but also apply library size normalization (*100000) to ground truth (df1)
        mat1 <- as.matrix(df1)
        lib_sizes <- rowSums(mat1)
        lib_sizes[lib_sizes == 0] <- 1  # avoid division by zero
        mat1 <- mat1 / lib_sizes * 100000
        mat1 <- apply_trans(mat1, this_trans_factor)
        df1 <- as.data.frame(mat1)

        cat(paste("data_mode: v1_trans_v2_trans_norm_100000 | ground truth library size norm *100000 + trans:", this_trans_factor, "\n"))
    }
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

    # Add data_mode info to subtitle
    if (data_mode != "default") {
        subtitle_text_with_stats <- paste0(
            subtitle_text_with_stats,
            "\n| data_mode: ", data_mode
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

    # Add data_mode tag to file prefix
    if (data_mode != "default") {
        file_prefix <- paste0(file_prefix, data_mode, "_")
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
                            if (filter_zero_gt) " [GT-zero filtered]" else "",
                            if (data_mode != "default") paste0(" [", data_mode, "]") else ""),
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
        data_mode = data_mode,
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
                            if (data_mode != "default") paste0("_", data_mode) else "",
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

# 1. Pearson, unfiltered
tryCatch(
    run_correlation_scatter(
        path1 = path1, path2 = path2, path3 = path3,
        out_dir = out_dir, saved_models_dir = saved_models_dir,
        factor = factor, V2_norm_factor = V2_norm_factor,
        this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
        save_csv_dir = save_csv_dir, corr = corr,
        y_title = y_title, x_title = x_title,
        corr_method = "pearson", filter_zero_gt = FALSE,
        data_mode = data_mode
    ),
    error = function(e) cat("ERROR in Pearson unfiltered:", conditionMessage(e), "\n")
)

# 3. Spearman, unfiltered
tryCatch(
    run_correlation_scatter(
        path1 = path1, path2 = path2, path3 = path3,
        out_dir = out_dir, saved_models_dir = saved_models_dir,
        factor = factor, V2_norm_factor = V2_norm_factor,
        this_trans_factor = this_trans_factor, V2_trans_factor = V2_trans_factor,
        save_csv_dir = save_csv_dir, corr = corr,
        y_title = y_title, x_title = x_title,
        corr_method = "spearman", filter_zero_gt = FALSE,
        data_mode = data_mode
    ),
    error = function(e) cat("ERROR in Spearman unfiltered:", conditionMessage(e), "\n")
)

