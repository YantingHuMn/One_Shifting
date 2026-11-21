library(readr)
library(dplyr)
library(ggplot2)
library(patchwork)

args <- commandArgs(trailingOnly = TRUE)

data_path <- args[1]
png_file <- args[2]
method <- args[3]
sample_name <- args[4]
train_method <- args[5]

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
                "count+1",
                "sqrt",
                "sqrt+1",
                "log2",
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
    "Inverse-transformed normalized counts | ", train_method, "\n",
    "\u0394r = r_recon \u2212 r_input; \u0394r > 0 indicates improvement | n = ",
    nrow(df)
)

p <- ggplot(df, aes(x = recon_mean, y = negative_residual_mean, color = norm, shape = trans)) +
    geom_hline(yintercept = 0, color = "grey65", linewidth = 0.5, linetype = "dashed") +
    geom_vline(xintercept = 0, color = "grey65", linewidth = 0.5, linetype = "dashed") +
    geom_point(size = 4, stroke = 1, alpha = 0.7) +
    scale_color_manual(
        values = c(
            "no_norm" = "#FFA660",
            "1000000" = "#D3B0DE",
            "100000" = "#A7B8DB",
            "10000" = "#ACD1BE",
            "1000" = "#DBA3B4",
            "standardize" = "#D85B59"
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
    # three pairs share a shape: hollow = no shifting, filled = one shifting
    scale_shape_manual(
        values = c(
            "no_trans" = 1,
            "count+1" = 16,
            "sqrt" = 0,
            "sqrt+1" = 15,
            "log2" = 2,
            "log2(count+2)" = 17
        ),
        name = "Transformation"
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

# Zoom window: the cluster at the top right. Sort recon_mean and cut at the largest gap
# in the upper half; optional args 5-8 (xmin xmax ymin ymax) set the window by hand.
if (length(args) >= 8) {
    zoom_x <- as.numeric(args[5:6])
    zoom_y <- as.numeric(args[7:8])
} else {
    xs <- sort(df$recon_mean)
    upper <- seq(ceiling(length(xs) / 2), length(xs) - 1)
    cut_at <- upper[which.max(diff(xs)[upper])]
    top <- df$recon_mean > xs[cut_at]
    pad_x <- 0.1 * diff(range(df$recon_mean[top])) + 0.005
    pad_y <- 0.1 * diff(range(df$negative_residual_mean[top])) + 0.005
    zoom_x <- range(df$recon_mean[top]) + c(-pad_x, pad_x)
    zoom_y <- range(df$negative_residual_mean[top]) + c(-pad_y, pad_y)
}

# Left: the full plot with the zoom window boxed. Middle: a narrow strip carrying the two
# connector lines. Right: the zoom window itself. patchwork aligns the three panels, so the
# lines run straight from the box corners to the corners of the zoomed panel.
full_range <- ggplot_build(p)$layout$panel_params[[1]]
xr <- full_range$x.range
yr <- full_range$y.range
gap <- 0.25                                   # width of the connector strip, in panel widths

to_npc_y <- function(y) (y - yr[1]) / diff(yr)
box_x <- (zoom_x[2] - xr[1]) / diff(xr)       # right edge of the box, in left-panel npc
line_ends <- function(y_box, y_zoom) {
    # straight line from (box_x, y_box) to (1 + gap, y_zoom); value at the left panel edge
    y_edge <- y_box + (y_zoom - y_box) * (1 - box_x) / (1 + gap - box_x)
    c(y_box = y_box, y_edge = y_edge, y_zoom = y_zoom)
}
top <- line_ends(to_npc_y(zoom_y[2]), 1)
bottom <- line_ends(to_npc_y(zoom_y[1]), 0)
from_npc_y <- function(v) yr[1] + v * diff(yr)

p_left <- p +
    annotate(
        "rect",
        xmin = zoom_x[1], xmax = zoom_x[2],
        ymin = zoom_y[1], ymax = zoom_y[2],
        fill = NA, color = "grey35", linewidth = 0.5
    ) +
    annotate(
        "segment",
        x = zoom_x[2], xend = xr[2],
        y = c(zoom_y[2], zoom_y[1]),
        yend = from_npc_y(c(top["y_edge"], bottom["y_edge"])),
        color = "grey50", linewidth = 0.4
    ) +
    coord_cartesian(xlim = xr, ylim = yr, expand = FALSE) +
    labs(title = NULL, subtitle = NULL) +
    theme(legend.position = "none", plot.margin = margin(8, 0, 6, 8))

p_link <- ggplot() +
    annotate(
        "segment",
        x = 0, xend = 1,
        y = c(top["y_edge"], bottom["y_edge"]),
        yend = c(top["y_zoom"], bottom["y_zoom"]),
        color = "grey50", linewidth = 0.4
    ) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    theme_void() +
    theme(plot.margin = margin(0, 0, 0, 0))

p_zoom <- p +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0), position = "right") +
    coord_cartesian(xlim = zoom_x, ylim = zoom_y, expand = FALSE) +
    labs(title = NULL, subtitle = NULL) +
    theme(plot.title = element_text(size = 12, face = "bold"), plot.margin = margin(8, 8, 6, 0))

# title and subtitle span the whole figure instead of being clipped to the left panel
p_side <- p_left + p_link + p_zoom + plot_layout(widths = c(1, gap, 1)) +
    plot_annotation(
        title = p$labels$title,
        subtitle = p$labels$subtitle,
        theme = theme(
            plot.title = element_text(size = 15, face = "bold", margin = margin(b = 4)),
            plot.subtitle = element_text(size = 9.5, color = "grey25", margin = margin(b = 8))
        )
    )

ggsave(png_file, p_side, width = 14, height = 5, units = "in", dpi = 400, bg = "white")

cat("Saved:\n", png_file, "\n")