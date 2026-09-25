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
    "\u0394r = r_input \u2212 r_recon; \u2212\u0394r > 0 indicates improvement | n = ",
    nrow(df)
)

p <- ggplot(
    df,
    aes(
        x = recon_mean,
        y = negative_residual_mean,
        color = trans,
        shape = norm
    )
) +
    geom_hline(
        yintercept = 0,
        color = "grey65",
        linewidth = 0.5,
        linetype = "dashed"
    ) +
    geom_vline(
        xintercept = 0,
        color = "grey65",
        linewidth = 0.5,
        linetype = "dashed"
    ) +
    geom_point(
        size = 4,
        stroke = 1,
        alpha = 0.7
    ) +
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
        expand = expansion(mult = 0.05)
    ) +
    scale_y_continuous(
        expand = expansion(mult = 0.05)
    ) +
    labs(
        title = paste0("Mean Gene-wise ", method, " Performance for ", sample_name),
        subtitle = subtitle_text,
        x = expression(Mean~r[recon]),
        y = expression(Mean~-Delta*r~"("*r[recon] - r[input]*")")
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
        panel.grid.minor = element_line(
            color = "grey94",
            linewidth = 0.3
        ),
        plot.margin = margin(8, 8, 6, 8)
    )

dir.create(
    dirname(png_file),
    recursive = TRUE,
    showWarnings = FALSE
)

ggsave(
    png_file,
    p,
    width = 8,
    height = 5,
    units = "in",
    dpi = 400,
    bg = "white"
)

cat("Saved:\n", png_file, "\n")