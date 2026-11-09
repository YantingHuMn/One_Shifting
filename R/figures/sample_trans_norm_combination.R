library(readr)
library(dplyr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

data_path <- args[1]
png_file <- args[2]
method <- args[3]
sample_name <- args[4]

df <- read_csv(data_path, show_col_types = FALSE) %>%
    transmute(
        trans = REP1_trans,
        norm = REP1_norm,
        recon_mean = as.numeric(x_mean),
        negative_residual_mean = -as.numeric(residual_mean)
    ) %>%
    filter(
        is.finite(recon_mean),
        is.finite(negative_residual_mean)
    ) %>%
    mutate(
        trans = factor(
            trans,
            levels = c(
                "no_trans",
                "sqrt",
                "sqrt+1",
                "log2",
                "count+1",
                "log2(count+2)"
            )
        ),
        norm = factor(
            norm,
            levels = c(
                "no_norm",
                "1000000",
                "100000",
                "10000",
                "1000",
                "standardize"
            )
        )
    )

subtitle_text <- paste0(
    "Inverse-transformed normalized counts | VAE", "\n",
    "\u0394r = r_recon \u2212 r_input; \u0394r > 0 indicates improvement | n = ",
    nrow(df)
)

p <- ggplot(df, aes(x = recon_mean, y = negative_residual_mean, color = trans, shape = norm)) +
    geom_hline(yintercept = 0, color = "grey65", linewidth = 0.5, linetype = "dashed") +
    geom_vline(xintercept = 0, color = "grey65", linewidth = 0.5, linetype = "dashed") +
    geom_point(size = 4, stroke = 1, alpha = 0.7) +
    scale_color_manual(
        values = c(
            "no_trans" = "#F8766D",
            "sqrt" = "#70ACDE",
            "sqrt+1" = "#7FC97F",
            "log2" = "#BEAED4",
            "count+1" = "#FFA660",
            "log2(count+2)" = "#C58C6B"
        ),
        name = "Transformation"
    ) +
    scale_shape_manual(
        values = c(
            "no_norm" = 16,
            "1000000" = 17,
            "100000" = 15,
            "10000" = 18,
            "1000" = 6,
            "standardize" = 1
        ),
        labels = c(
            "No normalization",
            "1 million",
            "100,000",
            "10,000",
            "1,000",
            "Standardization"
        ),
        name = "Normalization"
    ) +
    scale_x_continuous(
        breaks = function(limits) {
            seq(ceiling(limits[1] * 10), floor(limits[2] * 10)) / 10
        },
        expand = expansion(mult = 0.05)
    ) +
    scale_y_continuous(
        breaks = function(limits) {
            seq(ceiling(limits[1] * 10), floor(limits[2] * 10)) / 10
        },
        expand = expansion(mult = 0.05)
    ) +
    labs(
        title = paste0("Mean Gene-wise ", method, " Performance for ", sample_name),
        subtitle = subtitle_text,
        x = expression(Mean~r[recon]),
        y = expression(Mean~"("*r[recon] - r[input]*")")
    ) +
    guides(
        color = guide_legend(order = 1),
        shape = guide_legend(order = 2)
    ) +
    theme_bw(base_size = 12) +
    theme(
        plot.title = element_text(
            size = 15,
            face = "bold",
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 9.5,
            color = "grey25",
            margin = margin(b = 8)
        ),
        axis.title = element_text(size = 11),
        legend.title = element_text(
            size = 10,
            face = "bold"
        ),
        legend.text = element_text(size = 9),
        legend.key.height = unit(0.55, "cm"),
        panel.grid.major = element_line(
            color = "grey88",
            linewidth = 0.4
        ),
        panel.grid.minor = element_blank(),
        plot.margin = margin(8, 8, 6, 8)
    )

dir.create(
    dirname(png_file),
    recursive = TRUE,
    showWarnings = FALSE
)

original_png_file <- paste0(tools::file_path_sans_ext(png_file), "_original.png")

ggsave(original_png_file, p, width = 8, height = 5, units = "in", dpi = 400, bg = "white")

cat("Saved:\n", original_png_file, "\n")

p_zoom <- p +
    coord_fixed(
        ratio = 1,
        xlim = c(0.20, 0.50),
        ylim = c(0, 0.20),
        expand = FALSE
    ) +
    labs(
        title = NULL,
        subtitle = NULL,
        x = NULL,
        y = NULL
    ) +
    theme(
        legend.position = "none",
        axis.text = element_text(size = 8),
        plot.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(3, 3, 3, 3)
    )

p <- p +
    annotate(
        "rect",
        xmin = 0.20, xmax = 0.50,
        ymin = 0, ymax = 0.20,
        fill = NA, color = "grey35", linewidth = 0.5
    )

panel <- ggplot_build(p)$layout$panel_params[[1]]
x_range <- panel$x.range
y_range <- panel$y.range

inset_xmin <- x_range[1] + 0.01 * diff(x_range)
inset_xmax <- 0.05
inset_ymin <- y_range[2] - 0.48 * diff(y_range)
inset_ymax <- y_range[2] - 0.01 * diff(y_range)

p <- p +
    coord_cartesian(
        xlim = x_range,
        ylim = y_range,
        expand = FALSE
    ) +
    annotate(
        "segment",
        x = inset_xmax, y = inset_ymax,
        xend = 0.20, yend = 0.20,
        color = "grey50", linewidth = 0.4
    ) +
    annotate(
        "segment",
        x = inset_xmax, y = inset_ymin,
        xend = 0.20, yend = 0,
        color = "grey50", linewidth = 0.4
    ) +
    annotate(
        "rect",
        xmin = inset_xmin, xmax = inset_xmax,
        ymin = inset_ymin, ymax = inset_ymax,
        fill = "white", color = NA
    ) +
    annotation_custom(
        grob = ggplotGrob(p_zoom),
        xmin = inset_xmin,
        xmax = inset_xmax,
        ymin = inset_ymin,
        ymax = inset_ymax
    ) +
    annotate(
        "rect",
        xmin = inset_xmin, xmax = inset_xmax,
        ymin = inset_ymin, ymax = inset_ymax,
        fill = NA, color = "grey35", linewidth = 0.5
    )

ggsave(png_file, p, width = 8, height = 5, units = "in", dpi = 400, bg = "white")

cat("Saved:\n", png_file, "\n")