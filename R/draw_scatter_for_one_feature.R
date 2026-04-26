
draw_scatter_for_one_feature <- function(input_path, recon_path, feature, out_dir = NULL) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(arrow)
        library(ggplot2)
    })

    # Helper: accept either a path (string) or a data.frame
    read_input <- function(x) {
        if (is.data.frame(x)) return(x)
        read_feather(x)
    }

    input_df <- read_input(input_path)
    recon_df <- read_input(recon_path)

    # Default out_dir: use dirname of whichever arg is a path
    if (is.null(out_dir)) {
        if (is.character(recon_path)) {
            out_dir <- dirname(recon_path)
        } else if (is.character(input_path)) {
            out_dir <- dirname(input_path)
        } else {
            out_dir <- getwd()
        }
    }

    if (!dir.exists(out_dir)) {
        dir.create(out_dir, recursive = TRUE)
    }

    y_vals <- input_df[[feature]]
    x_vals <- recon_df[[feature]]

    n <- min(length(x_vals), length(y_vals))
    plot_df <- data.frame(recon = x_vals[1:n], input = y_vals[1:n])

    p <- ggplot(plot_df, aes(x = recon, y = input)) +
        geom_point(alpha = 0.5, size = 1) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
        labs(
        x = paste0("Reconstructed (", feature, ")"),
        y = paste0("Input (", feature, ")"),
        title = paste0("Input vs Reconstructed: ", feature)
        ) +
        theme_minimal()

    out_file <- file.path(out_dir, paste0("scatter_", feature, ".png"))
    ggsave(out_file, p, width = 6, height = 6, dpi = 150)

    message("Saved to: ", out_file)
    invisible(p)
}