library(readr)
library(dplyr)
library(ggplot2)
library(hexbin)

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
png_file <- args[2]
method <- args[3]
norm <- args[4]
trans <- args[5]
sample_name <- args[6]

df <- read_csv(data_path, show_col_types = FALSE) %>%
    transmute(
        gene = Value,
        x = as.numeric(VAE),
        y = as.numeric(ORIG)
    ) %>%
    filter(is.finite(x), is.finite(y)) %>%
    mutate(
        residual = y - x,
        position = case_when(
            residual > 0 ~ "above",
            residual < 0 ~ "below",
            TRUE ~ "on"
        )
    )

x_mean <- mean(df$x)
residual_mean <- mean(df$residual)
weighted_residual_mean <- mean(df$x * df$residual)

n_above <- sum(df$position == "above")
n_below <- sum(df$position == "below")
n_on <- sum(df$position == "on")
below_percentage <- 100 * n_below / nrow(df)

subtitle_text <- paste0(
    "Preprocessing: library-size normalization to ", norm, " followed by ", trans, "\n",
    "Evaluation: inverse-transformed normalized counts\n",
    "Mean x: ", sprintf("%.4f", x_mean),
    "  |  Mean residual: ", sprintf("%.4f", residual_mean),
    "  |  Mean weighted residual: ",
    sprintf("%.4f", weighted_residual_mean), "\n",
    "Above the 45\u00B0 line: ", format(n_above, big.mark = ","),
    "  |  Below the 45\u00B0 line: ", format(n_below, big.mark = ","), "(", sprintf("%.2f%%", below_percentage), ")",
    "  |  On line: ", format(n_on, big.mark = ",")
)


p <- ggplot(df, aes(x = x, y = y)) +
    geom_hex(bins = 60) +
    scale_x_continuous(limits = c(-0.1, 0.5)) +
    scale_y_continuous(limits = c(-0.1, 0.5)) + 
    geom_abline(
        slope = 1,
        intercept = 0,
        color = "grey35",
        linewidth = 0.7,
        linetype = "dashed"
    ) +
    scale_fill_gradientn(
        colours = c(
            "#5271AE",
            "#70ACDE",
            "#F5CC7D",
            "#FFA660",
            "#D85B59"
        ),
        name = "Gene count"
    ) +
    coord_fixed(
        ratio = 1,
        expand = FALSE
    ) +
    labs(
        title = paste0("Gene-wise ", method, " Correlations for ", sample_name),
        subtitle = subtitle_text,
        x = expression(r[recon]),
        y = expression(r[input])
    ) +
    theme_classic(base_size = 12) +
    theme(
        plot.title = element_text(
            size = 15,
            face = "bold",
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 9.5,
            lineheight = 1.15,
            color = "grey25",
            margin = margin(b = 8)
        ),
        axis.title = element_text(size = 11),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 9),
        legend.key.height = unit(1.4, "cm"),
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
    width = 6.7,
    height = 5.8,
    units = "in",
    dpi = 400,
    bg = "white"
)

cat("Saved:\n", png_file, "\n")