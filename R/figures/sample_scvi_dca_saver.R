library(readr)
library(dplyr)
library(ggplot2)
library(hexbin)

args <- commandArgs(trailingOnly = TRUE)

transformer_path <- args[1]
scvi_path <- args[2]
dca_path <- args[3]
saver_path <- args[4]
png_file <- args[5]
correlation_type <- args[6]
sample_name <- args[7]

transformer_df <- read_csv(transformer_path, show_col_types = FALSE) %>%
    transmute(gene = Value, x = as.numeric(VAE))

scvi_df <- read_csv(scvi_path, show_col_types = FALSE) %>%
    transmute(gene = Value, y = as.numeric(VAE)) %>%
    inner_join(transformer_df, by = "gene") %>%
    mutate(method = "scVI")

dca_df <- read_csv(dca_path, show_col_types = FALSE) %>%
    transmute(gene = Value, y = as.numeric(VAE)) %>%
    inner_join(transformer_df, by = "gene") %>%
    mutate(method = "DCA")

saver_df <- read_csv(saver_path, show_col_types = FALSE) %>%
    transmute(gene = Value, y = as.numeric(VAE)) %>%
    inner_join(transformer_df, by = "gene") %>%
    mutate(method = "SAVER")

df <- bind_rows(scvi_df, dca_df, saver_df) %>%
    filter(is.finite(x), is.finite(y)) %>%
    mutate(method = factor(method, levels = c("scVI", "DCA", "SAVER")))

panel_stats <- df %>%
    group_by(method) %>%
    summarise(
        total = n(),
        above = sum(y > x),
        below = sum(y < x),
        on_line = sum(y == x),
        .groups = "drop"
    ) %>%
    mutate(
        label = sprintf(
            "%s\nAbove: %d (%.1f%%) \n Below: %d (%.1f%%)",
            method,
            above, 100 * above / total,
            below, 100 * below / total,
            on_line, 100 * on_line / total
        )
    )

panel_labels <- setNames(panel_stats$label, as.character(panel_stats$method))

p <- ggplot(df, aes(x = x, y = y)) +
    geom_hex(bins = 60) +
    geom_abline(
        slope = 1, intercept = 0,
        color = "grey35", linewidth = 0.7, linetype = "dashed"
    ) +
    scale_fill_gradientn(
        colours = c("#5271AE", "#70ACDE", "#F5CC7D", "#FFA660", "#D85B59"),
        name = "Gene count"
    ) +
    facet_wrap(
        ~method, nrow = 1,
        labeller = labeller(method = as_labeller(panel_labels))
    ) +
    coord_fixed(ratio = 0.5) +
    labs(
        title = paste0("Gene-wise Output ", correlation_type, " Correlations for ", sample_name),
        x = "Transformer output correlation",
        y = "Baseline output correlation"
    ) +
    theme_classic(base_size = 12) +
    theme(
        plot.title = element_text(size = 15, face = "bold", margin = margin(b = 8)),
        strip.text = element_text(size = 9, face = "bold", lineheight = 1.1),
        axis.title = element_text(size = 11),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 9),
        legend.key.height = unit(1.4, "cm"),
        plot.margin = margin(8, 8, 6, 8)
    )

dir.create(dirname(png_file), recursive = TRUE, showWarnings = FALSE)

ggsave(
    png_file, p,
    width = 10, height = 4,
    units = "in", dpi = 400, bg = "white"
)

cat("Saved:\n", png_file, "\n")
print(panel_stats)