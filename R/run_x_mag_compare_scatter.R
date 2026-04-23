source("../One_Shifting/R/create_hex_plot.R")

run_x_mag_compare_scatter <- function(x_dir, y_dir, V1, V2, x_norm_info_path, corr_dir,
                                       v2_trans_factor = "no_trans",
                                       v2_norm_factor = "no_norm",
                                       v1_trans_factors = c("no_trans", "sqrt", "log2", "count+1", "sqrt+1", "log2(count+2)")) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(dplyr)
        library(ggplot2)
        library(gridExtra)
        library(viridis)
    })

    x_label <- basename(x_dir)
    if (grepl("^given_", x_label)) x_label <- basename(dirname(x_dir))
    y_label <- basename(y_dir)
    if (grepl("^given_", y_label)) y_label <- basename(dirname(y_dir))

    out_dir <- file.path(y_dir, paste0("x_mag_compare_", x_label, "_VS_", y_label))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    if (corr_dir == "col") {
        object = "gene"
    } else {
        object = "cell"
    }

    # ==================== Pearson plots ====================
    corr = "pearson"
    x_norm_info_path <- paste0(x_dir, "/plots_summary_", corr, "_", corr_dir, "_", object, "_best_norm.csv")
    y_norm_info_path <- paste0(y_dir, "/plots_summary_", corr, "_", corr_dir, "_", object, "_best_norm.csv")

    best_norm_x <- if (file.exists(x_norm_info_path)) read.csv(x_norm_info_path) else NULL
    best_norm_y <- if (file.exists(y_norm_info_path)) read.csv(y_norm_info_path) else NULL

    plots_density <- list()

    for (trans_val in v1_trans_factors) {
        cat("Processing trans: ", trans_val, "\n")

        if (!is.null(best_norm_x)) {
            v1_norm_factor_x <- best_norm_x$REP1_norm[best_norm_x$REP1_trans == trans_val]
            if (length(v1_norm_factor_x) == 0) {
                warning(paste("No norm found in x for trans=", trans_val))
                next
            }
            path1 <- file.path(x_dir, paste0("Figures_", corr_dir, "/d_pearson_", corr_dir, "_scatter_v1_trans_", trans_val, "_norm_", v1_norm_factor_x, "_v2_trans_", v2_trans_factor, "_norm_", v2_norm_factor, ".csv"))
            v1_norm_factor <- v1_norm_factor_x
        } else {
            path1 <- file.path(x_dir, paste0("Figures_", corr_dir, "/d_pearson_", corr_dir, "_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv"))
            v1_norm_factor <- "no_norm"
        }

        if (!is.null(best_norm_y)) {
            v1_norm_factor_y <- best_norm_y$REP1_norm[best_norm_y$REP1_trans == trans_val]
            if (length(v1_norm_factor_y) == 0) {
                warning(paste("No norm found in y for trans=", trans_val))
                next
            }
            path2 <- file.path(y_dir, paste0("Figures_", corr_dir, "/d_pearson_", corr_dir, "_scatter_v1_trans_", trans_val, "_norm_", v1_norm_factor_y, "_v2_trans_", v2_trans_factor, "_norm_", v2_norm_factor, ".csv"))
        } else {
            path2 <- file.path(y_dir, paste0("Figures_", corr_dir, "/d_pearson_", corr_dir, "_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv"))
        }

        cat("      with norm: ", v1_norm_factor, "\n")

        if (!file.exists(path1) || !file.exists(path2)) {
            warning(paste("One of the paths does not exist:", path1, "or", path2))
            next  
        }

        df1 <- read.csv(path1)
        df2 <- read.csv(path2)

        df2_sorted <- df2[match(df1$Value, df2$Value), ]

        df1_x_mean <- mean(df1$VAE, na.rm = TRUE)
        df2_y_mean <- mean(df2_sorted$VAE, na.rm = TRUE)

        plot_data <- data.frame(
            Value = df1$Value,
            df1_VAE = df1$VAE,
            df2_VAE = df2_sorted$VAE
        )

        residuals <- plot_data$df2_VAE - plot_data$df1_VAE
        residual_sum <- sum(residuals, na.rm = TRUE)
        residual_mean <- mean(residuals, na.rm = TRUE)

        above_line <- sum(plot_data$df2_VAE > plot_data$df1_VAE, na.rm = TRUE)
        below_line <- sum(plot_data$df2_VAE < plot_data$df1_VAE, na.rm = TRUE)
        on_line <- sum(plot_data$df2_VAE == plot_data$df1_VAE, na.rm = TRUE)

        n_points <- nrow(plot_data)
        
        plots_density[[trans_val]] <- create_hex_plot(plot_data, df1_x_mean, df2_y_mean, 
                                                       residual_sum, residual_mean, n_points, 
                                                       above_line, below_line, on_line, 
                                                       trans_val, v1_norm_factor, "Pearson", V1, y_label, x_label)
    }

    # Save
    if (length(plots_density) > 0) {
        grid_plot <- do.call(grid.arrange, c(plots_density, ncol = 3))
        ggsave(file.path(out_dir, paste0("pearson_density_", corr_dir, ".png")), grid_plot, width = 15, height = 10, dpi = 300)
    }

    # ==================== Spearman plots ====================
    corr = "spearman"
    x_norm_info_path <- paste0(x_dir, "/plots_summary_", corr, "_", corr_dir, "_", object, "_best_norm.csv")
    y_norm_info_path <- paste0(y_dir, "/plots_summary_", corr, "_", corr_dir, "_", object, "_best_norm.csv")

    best_norm_x <- if (file.exists(x_norm_info_path)) read.csv(x_norm_info_path) else NULL
    best_norm_y <- if (file.exists(y_norm_info_path)) read.csv(y_norm_info_path) else NULL

    plots_density <- list()

    for (trans_val in v1_trans_factors) {
        cat("Processing trans: ", trans_val, "\n")

        if (!is.null(best_norm_x)) {
            v1_norm_factor_x <- best_norm_x$REP1_norm[best_norm_x$REP1_trans == trans_val]
            if (length(v1_norm_factor_x) == 0) {
                warning(paste("No norm found in x for trans=", trans_val))
                next
            }
            path1 <- file.path(x_dir, paste0("Figures_", corr_dir, "/g_spearman_", corr_dir, "_scatter_v1_trans_", trans_val, "_norm_", v1_norm_factor_x, "_v2_trans_", v2_trans_factor, "_norm_", v2_norm_factor, ".csv"))
            v1_norm_factor <- v1_norm_factor_x
        } else {
            path1 <- file.path(x_dir, paste0("Figures_", corr_dir, "/g_spearman_", corr_dir, "_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv"))
            v1_norm_factor <- "no_norm"
        }

        if (!is.null(best_norm_y)) {
            v1_norm_factor_y <- best_norm_y$REP1_norm[best_norm_y$REP1_trans == trans_val]
            if (length(v1_norm_factor_y) == 0) {
                warning(paste("No norm found in y for trans=", trans_val))
                next
            }
            path2 <- file.path(y_dir, paste0("Figures_", corr_dir, "/g_spearman_", corr_dir, "_scatter_v1_trans_", trans_val, "_norm_", v1_norm_factor_y, "_v2_trans_", v2_trans_factor, "_norm_", v2_norm_factor, ".csv"))
        } else {
            path2 <- file.path(y_dir, paste0("Figures_", corr_dir, "/g_spearman_", corr_dir, "_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv"))
        }

        cat("      with norm: ", v1_norm_factor, "\n")

        if (!file.exists(path1) || !file.exists(path2)) {
            warning(paste("One of the paths does not exist:", path1, "or", path2))
            next  
        }

        df1 <- read.csv(path1)
        df2 <- read.csv(path2)

        df2_sorted <- df2[match(df1$Value, df2$Value), ]

        df1_x_mean <- mean(df1$VAE, na.rm = TRUE)
        df2_y_mean <- mean(df2_sorted$VAE, na.rm = TRUE)

        plot_data <- data.frame(
            Value = df1$Value,
            df1_VAE = df1$VAE,
            df2_VAE = df2_sorted$VAE
        )

        residuals <- plot_data$df2_VAE - plot_data$df1_VAE
        residual_sum <- sum(residuals, na.rm = TRUE)
        residual_mean <- mean(residuals, na.rm = TRUE)

        above_line <- sum(plot_data$df2_VAE > plot_data$df1_VAE, na.rm = TRUE)
        below_line <- sum(plot_data$df2_VAE < plot_data$df1_VAE, na.rm = TRUE)
        on_line <- sum(plot_data$df2_VAE == plot_data$df1_VAE, na.rm = TRUE)

        n_points <- nrow(plot_data)
        
        plots_density[[trans_val]] <- create_hex_plot(plot_data, df1_x_mean, df2_y_mean, 
                                                       residual_sum, residual_mean, n_points, 
                                                       above_line, below_line, on_line, 
                                                       trans_val, v1_norm_factor, "Spearman", V1, y_label, x_label)
    }

    # Save
    if (length(plots_density) > 0) {
        grid_plot <- do.call(grid.arrange, c(plots_density, ncol = 3))
        ggsave(file.path(out_dir, paste0("spearman_density_", corr_dir, ".png")), grid_plot, width = 15, height = 10, dpi = 300)
    }

    cat("Done! Saved 2 plots (1 Pearson + 1 Spearman) to:", out_dir, "\n")
}


args <- commandArgs(trailingOnly = TRUE)

run_x_mag_compare_scatter(
    x_dir = args[1],
    y_dir = args[2],
    V1 = args[3],
    V2 = args[4],
    x_norm_info_path = args[5],
    corr_dir = args[6]
)