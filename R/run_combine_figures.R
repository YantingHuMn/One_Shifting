args <- commandArgs(trailingOnly = TRUE)

input_dir <- args[1]
col_or_row <- args[2]
# way <- as.character(args[3])

if (is.na(col_or_row) || col_or_row == "") {
  col_or_row <- "col"
}

source("../One_Shifting/R/extract_title.R")
source("../One_Shifting/R/create_no_data_plot.R")

run_combine_figures <- function(input_dir, way, num, col_or_row = "col") {
    # Load libraries
    suppressPackageStartupMessages({
        library(png)
        library(ggplot2)
        library(gridExtra)
        library(grid)
    })
        
    # Output file paths
    pearson_output_file <- file.path(dirname(input_dir), paste0("AAA_pearson_scatter_combine_", col_or_row, ".png"))
    spearman_output_file <- file.path(dirname(input_dir), paste0("AAA_spearman_scatter_combine_", col_or_row, ".png"))
    roc_output_file <- file.path(dirname(input_dir), paste0("AAA_roc_curves_combine_", col_or_row, ".png"))
    roc_balance_output_file <- file.path(dirname(input_dir), paste0("AAA_roc_balance_curves_combine_", col_or_row, ".png"))

    # Find matching PNG files
    if (way == "pearson") {
        png_files1 <- list.files(input_dir, pattern = paste0("^d_pearson_", col_or_row, "_scatter_.*\\.png$"), full.names = TRUE)
        png_files2 <- list.files(input_dir, pattern = "^d_pearson_scatter_.*\\.png$", full.names = TRUE)
        png_files <- sort(unique(c(png_files1, png_files2)))
    } else if (way == "spearman") {
        png_files1 <- list.files(input_dir, pattern = paste0("^g_spearman_", col_or_row, "_scatter_.*\\.png$"), full.names = TRUE)
        png_files2 <- list.files(input_dir, pattern = "^g_spearman_scatter_.*\\.png$", full.names = TRUE)
        png_files <- sort(unique(c(png_files1, png_files2)))
    } else if (way == "roc") {
        png_files <- sort(list.files(input_dir, pattern = "^j_roc_curves_.*\\.png$", full.names = TRUE))
    } else if (way == "roc_balanced") {
        png_files <- sort(list.files(input_dir, pattern = "^j_roc_curves_balanced_.*\\.png$", full.names = TRUE))
    }

    num_found <- length(png_files)
    cat("Found", num_found, "figures in", input_dir, "\n")

    # Extract unique transformations and norms
    transformation_data <- unique(lapply(png_files, extract_title))

    unique_transforms <- unique(sapply(transformation_data, function(x) {
        match <- regmatches(x, regexpr("V1: ([^;]*)", x))
        if (length(match) > 0) return(trimws(sub("V1: ", "", match))) else return(NA)
    }))
    unique_transforms <- unique_transforms[!is.na(unique_transforms)]
    print(unique_transforms)

    unique_norms <- unique(sapply(transformation_data, function(x) {
        match <- regmatches(x, regexpr("norm (.*)", x))
        if (length(match) > 0) return(trimws(sub("norm ", "", match))) else return(NA)
    }))
    unique_norms <- unique_norms[!is.na(unique_norms)]
    unique_norms <- unique_norms[unique_norms != "no"]
    print(unique_norms)

    width1 <- 4 * length(unique_norms)
    height1 <- 4 * length(unique_transforms)

    # Build plot matrix
    plots <- matrix(list(), nrow = length(unique_transforms), ncol = length(unique_norms))

    for (i in seq_along(unique_transforms)) {
        for (j in seq_along(unique_norms)) {
        pattern <- paste0("v1_trans_", unique_transforms[i], "_norm_", unique_norms[j], "_v2")
        matched_files <- grep(pattern, png_files, value = TRUE, fixed = TRUE)

        if (length(matched_files) > 0) {
            img_path <- matched_files[1]
            img <- readPNG(img_path)
            plot_title <- extract_title(img_path)

            plots[[i, j]] <- ggplot() +
            annotation_raster(img, xmin = 0, xmax = 1, ymin = 0, ymax = 1) +
            xlim(0, 1) + ylim(0, 1) +
            ggtitle(plot_title) +
            theme_void() +
            theme(plot.title = element_text(hjust = 0.5, size = 8, margin = margin(b = 2)))
            print(plot_title)
        } else {
            plots[[i, j]] <- create_no_data_plot()
            print(paste("No data for transformation:", unique_transforms[i], "and norm:", unique_norms[j]))
        }
        }
    }

    plots_list <- as.vector(t(plots))

    # Combine and save
    if (way == "pearson") {
        combined <- do.call(grid.arrange, c(plots_list, list(ncol = length(unique_norms), top = textGrob("Pearson Scatter Correlation Analysis", gp = gpar(fontsize = 24, fontface = "bold")))))
        ggsave(pearson_output_file, combined, width = width1, height = height1, dpi = 300, bg = "white")
    } else if (way == "spearman") {
        combined <- do.call(grid.arrange, c(plots_list, list(ncol = length(unique_norms), top = textGrob("Spearman Scatter Correlation Analysis", gp = gpar(fontsize = 24, fontface = "bold")))))
        ggsave(spearman_output_file, combined, width = width1, height = height1, dpi = 300, bg = "white")
    } else if (way == "roc") {
        combined <- do.call(grid.arrange, c(plots_list, list(ncol = length(unique_norms), top = textGrob("ROC Comparison", gp = gpar(fontsize = 24, fontface = "bold")))))
        ggsave(roc_output_file, combined, width = width1, height = height1, dpi = 300, bg = "white")
    } else if (way == "roc_balanced") {
        combined <- do.call(grid.arrange, c(plots_list, list(ncol = length(unique_norms), top = textGrob("ROC Balanced Comparison", gp = gpar(fontsize = 24, fontface = "bold")))))
        ggsave(roc_balance_output_file, combined, width = width1, height = height1, dpi = 300, bg = "white")
    }
}

combine_figures(input_dir = input_dir, way = "pearson", num = num, col_or_row = col_or_row)
combine_figures(input_dir = input_dir, way = "spearman", num = num, col_or_row = col_or_row)