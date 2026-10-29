# Cross-sample comparison figure: published baselines vs one MSE arm
#
# Post: Builds two figures per correlation method, both with one row per sample and three
#   columns (scVI, DCA, SAVER).
#   The scatter figure plots the per-gene correlation-with-ground-truth of a baseline (x)
#   against that of the chosen MSE arm (y) at the arm's best (transformation,
#   normalization) for that sample. Points above the dashed y = x line are genes the MSE
#   arm reconstructs closer to the paired modality than the baseline.
#   The curve figure sweeps a correlation threshold t along the diagonal: at each t only
#   genes whose correlation is >= t for BOTH the baseline and the MSE arm are kept -- the
#   quadrant up and to the right of the point (t, t) -- and the curve is the percentage of
#   those genes lying above y = x. It answers how the advantage changes as attention is
#   restricted to progressively better-reconstructed genes.
#
# The best (trans, norm) is read per sample from
#   <method>/given_<V2>_no_norm_no_trans/Figures_col_v1_raw__v2_raw__recon_revtrans_revnorm/
#   plots_summary_<corr>_col_gene_v1_recon_reverse_trans_norm_best_norm.csv
# That file already holds the best norm WITHIN each transformation (select_best_norm.R
# groups by REP1_trans before slicing), and its combined_rank column is therefore constant
# across rows and carries no across-transformation information. The across-transformation
# choice is made here: highest x_mean and lowest residual_mean, combined by rank sum.
#
# Parameter:
#   base_dir     : One_Shifting_Results root, e.g. /dcs11/hongkai/data/yhu1/One_Shifting_Results
#   folder_name  : HCA, ENCODE or 10x
#   method       : the MSE arm to compare against, e.g. VAE, DCA_mse, Transformer_denoise
#   modality     : RNA or ATAC -- which modality is the model input
#   out_dir      : where the figures are written, e.g. <base_dir>/Paper_Figures
#   corr_methods : comma separated, default "pearson,spearman"
#   tsv_path     : sample table override; defaults to <base_dir>/<folder_name>_input_sample_pairs.tsv
#   panel_in     : panel side length in inches, default 2.0
#   min_n        : the curve is truncated once fewer than this many genes remain above the
#                  threshold, since the percentage becomes noise. Default 20.
# Output, per correlation method, into <out_dir> with the stem
#   <folder_name>_<method>_<modality>_<corr>_baseline_vs_<method>:
#   <stem>.png        the scatter grid
#   <stem>.csv        the per-panel numbers behind the scatter grid
#   <stem>_curve.png  the threshold-sweep grid
#   <stem>_curve.csv  percentage above y = x and genes retained at every threshold
# Row labels carry the selected transformation and normalization and, when
# <base_dir>/<folder_name>/<folder_name>_sparsity.csv is present, the input modality's
# sparsity for that sample.

suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
    stop("Usage: Rscript run_paper_figure_baseline_vs_method.R <base_dir> <folder_name> ",
         "<method> <modality> <out_dir> [corr_methods] [tsv_path] [panel_in]")
}
base_dir     <- args[1]
folder_name  <- args[2]
method       <- args[3]
modality     <- toupper(args[4])
out_dir      <- args[5]
corr_methods <- if (length(args) >= 6 && nzchar(args[6])) strsplit(args[6], ",")[[1]] else c("pearson", "spearman")
# tsv_path may be a single path or the comma separated list of every folder's table; in
# the latter case the entry whose file name starts with folder_name is used.
tsv_path <- if (length(args) >= 7 && nzchar(args[7])) {
    cand <- trimws(strsplit(args[7], ",")[[1]])
    cand <- cand[nzchar(cand)]
    if (length(cand) == 1) {
        cand
    } else {
        hit <- cand[startsWith(basename(cand), paste0(folder_name, "_"))]
        if (length(hit) != 1) {
            stop("Could not pick a sample table for folder_name '", folder_name,
                 "' out of: ", paste(basename(cand), collapse = ", "))
        }
        hit
    }
} else {
    file.path(base_dir, paste0(folder_name, "_input_sample_pairs.tsv"))
}
panel_in     <- if (length(args) >= 8 && nzchar(args[8])) as.numeric(args[8]) else 2.0
min_n        <- if (length(args) >= 9 && nzchar(args[9])) as.integer(args[9]) else 20L

if (!modality %in% c("RNA", "ATAC")) stop("modality must be RNA or ATAC, got: ", modality)

BASELINES <- c("scVI", "DCA", "SAVER")
DATA_MODE <- "v1_recon_reverse_trans_norm"
MODE_SUFFIX <- "v1_raw__v2_raw__recon_revtrans_revnorm"
V2_TRANS <- "no_trans"
V2_NORM  <- "no_norm"
AXIS_MIN <- -0.2
THRESHOLDS <- seq(AXIS_MIN, 1.0, by = 0.02)

# d_ for pearson, g_ for spearman -- matches file_prefix in both correlation scripts
prefix_for <- function(corr_method) {
    switch(corr_method,
           pearson  = "d_pearson_",
           spearman = "g_spearman_",
           stop("corr_method must be pearson or spearman, got: ", corr_method))
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------ sample table
if (!file.exists(tsv_path)) stop("Sample table not found: ", tsv_path)
tsv <- read.delim(tsv_path, stringsAsFactors = FALSE)
if (ncol(tsv) < 2) stop("Sample table needs at least two columns; column 2 holds sample_name: ", tsv_path)
samples <- as.character(tsv[[2]])
samples <- samples[nzchar(samples)]
if (length(samples) == 0) stop("No sample names in column 2 of ", tsv_path)
cat(sprintf("Samples from %s: %d\n", basename(tsv_path), length(samples)))

other_modality <- if (modality == "ATAC") "RNA" else "ATAC"

# ------------------------------------------------------------------ sparsity (optional)
# <folder>_sparsity.csv carries one row per (dataset, sample_name, data_type). The row for
# a sample is found by matching those three key columns by name -- never by row position,
# and never by assuming the input modality comes first. An absent file, an absent sample,
# or an unusable table costs the label only; the figures are still produced.
sparsity_of <- setNames(numeric(0), character(0))
sparsity_csv <- file.path(base_dir, folder_name, paste0(folder_name, "_sparsity.csv"))
KEY_COLS <- c("dataset", "sample_name", "data_type", "sparsity")

if (!file.exists(sparsity_csv)) {
    cat(sprintf("[warn] no sparsity table at %s; labels will omit sparsity\n", sparsity_csv))
} else {
    sp <- tryCatch(
        read.csv(sparsity_csv, stringsAsFactors = FALSE, row.names = NULL, check.names = FALSE),
        error = function(e) {
            cat(sprintf("[warn] could not read %s (%s); labels will omit sparsity\n",
                        basename(sparsity_csv), conditionMessage(e)))
            NULL
        })

    if (!is.null(sp) && !all(KEY_COLS %in% names(sp))) {
        cat(sprintf("[warn] %s lacks %s (has: %s); labels will omit sparsity\n",
                    basename(sparsity_csv),
                    paste(setdiff(KEY_COLS, names(sp)), collapse = " / "),
                    paste(names(sp), collapse = ", ")))
        sp <- NULL
    }

    if (!is.null(sp)) {
        ds <- toupper(trimws(as.character(sp$dataset)))
        dt <- toupper(trimws(as.character(sp$data_type)))
        sn <-         trimws(as.character(sp$sample_name))

        # When a data row carries a different number of fields than the header, read.csv
        # shifts every value one column over, so data_type ends up holding sample names.
        # Reject the table in that case rather than keying off misaligned values.
        if (!any(dt %in% c("RNA", "ATAC"))) {
            cat(sprintf("[warn] data_type in %s holds none of RNA / ATAC (saw: %s) -- the header and the data rows probably disagree on field count, so the columns are shifted; labels will omit sparsity\n",
                        basename(sparsity_csv), paste(utils::head(unique(dt), 4), collapse = ", ")))
        } else {
            hit <- which(dt == modality & ds == toupper(folder_name))
            if (length(hit) == 0) {
                cat(sprintf("[warn] %s has no dataset == %s and data_type == %s rows; labels will omit sparsity\n",
                            basename(sparsity_csv), folder_name, modality))
            } else {
                dup <- sn[hit][duplicated(sn[hit])]
                if (length(dup) > 0) {
                    cat(sprintf("[warn] %s has more than one %s / %s row for: %s -- the first of each is used\n",
                                basename(sparsity_csv), folder_name, modality,
                                paste(unique(dup), collapse = ", ")))
                    hit <- hit[!duplicated(sn[hit])]
                }
                sparsity_of <- setNames(as.numeric(sp$sparsity[hit]), sn[hit])
                cat(sprintf("Sparsity table: %d rows matched dataset == %s and data_type == %s in %s\n",
                            length(sparsity_of), folder_name, modality, basename(sparsity_csv)))
                missing_sp <- setdiff(samples, names(sparsity_of))
                if (length(missing_sp) > 0) {
                    cat(sprintf("[warn] no %s sparsity row for: %s\n",
                                modality, paste(missing_sp, collapse = ", ")))
                }
            }
        }
    }
}

sample_dir_of <- function(s) file.path(base_dir, folder_name, paste0(s, "_INPUT_", modality))
method_fig_dir_of <- function(s) {
    v2 <- paste0(s, "_", other_modality)
    file.path(sample_dir_of(s), method,
              paste0("given_", v2, "_", V2_NORM, "_", V2_TRANS),
              paste0("Figures_col_", MODE_SUFFIX))
}

# ------------------------------------------------------------------ best (trans, norm)
# Ranks across transformations: x_mean descending and residual_mean ascending, summed.
# Ties break on x_mean. Returns one row, plus whether the two criteria disagreed.
pick_best <- function(best_norm_csv) {
    df <- read.csv(best_norm_csv, stringsAsFactors = FALSE)
    needed <- c("REP1_trans", "REP1_norm", "x_mean", "residual_mean")
    if (!all(needed %in% names(df))) {
        stop("Columns ", paste(setdiff(needed, names(df)), collapse = ", "),
             " missing from ", best_norm_csv)
    }
    df <- df %>%
        mutate(
            x_rank   = rank(-x_mean,        ties.method = "min"),
            res_rank = rank(residual_mean,  ties.method = "min"),
            rank_sum = x_rank + res_rank
        ) %>%
        arrange(rank_sum, -x_mean)
    list(
        trans    = as.character(df$REP1_trans[1]),
        norm     = as.character(df$REP1_norm[1]),
        x_mean   = df$x_mean[1],
        residual = df$residual_mean[1],
        # TRUE when the highest x_mean and the lowest residual_mean are different rows
        split    = !identical(which.max(df$x_mean), which.min(df$residual_mean))
    )
}

# ------------------------------------------------------------------ gather
notes <- character(0)
panel_rows <- list()
stat_rows  <- list()

for (corr_method in corr_methods) {
    px <- prefix_for(corr_method)
    cat(sprintf("\n================ %s ================\n", corr_method))

    for (s in samples) {
        fig_dir <- method_fig_dir_of(s)
        bn_csv <- file.path(fig_dir, paste0("plots_summary_", corr_method,
                                            "_col_gene_", DATA_MODE, "_best_norm.csv"))
        if (!file.exists(bn_csv)) {
            notes <- c(notes, sprintf("[%s] %s: best_norm summary missing (%s)",
                                      corr_method, s, bn_csv))
            next
        }
        bb <- pick_best(bn_csv)
        if (bb$split) {
            notes <- c(notes, sprintf("[%s] %s: highest x_mean and lowest residual_mean are different transformations; took the rank sum (%s / %s)",
                                      corr_method, s, bb$trans, bb$norm))
        }

        method_csv <- file.path(
            fig_dir,
            paste0(px, DATA_MODE, "_col_scatter_v1_trans_", bb$trans,
                   "_norm_", bb$norm, "_v2_trans_", V2_TRANS, "_norm_", V2_NORM, ".csv"))
        if (!file.exists(method_csv)) {
            notes <- c(notes, sprintf("[%s] %s: scatter for the selected %s / %s missing (%s)",
                                      corr_method, s, bb$trans, bb$norm, basename(method_csv)))
            next
        }
        m <- read.csv(method_csv, stringsAsFactors = FALSE)
        if (!all(c("Value", "VAE", "ORIG") %in% names(m))) {
            notes <- c(notes, sprintf("[%s] %s: unexpected columns in %s", corr_method, s, basename(method_csv)))
            next
        }
        m <- m %>% select(gene = Value, y = VAE, orig_method = ORIG)

        for (bl in BASELINES) {
            bl_csv <- file.path(sample_dir_of(s), bl, "Figures_col",
                                paste0(px, "col_scatter_v1_trans_", V2_TRANS, "_norm_", V2_NORM,
                                       "_v2_trans_", V2_TRANS, "_norm_", V2_NORM, ".csv"))
            if (!file.exists(bl_csv)) {
                notes <- c(notes, sprintf("[%s] %s / %s: baseline scatter missing (%s)",
                                          corr_method, s, bl, bl_csv))
                next
            }
            b <- read.csv(bl_csv, stringsAsFactors = FALSE)
            if (!all(c("Value", "VAE", "ORIG") %in% names(b))) {
                notes <- c(notes, sprintf("[%s] %s / %s: unexpected columns in %s",
                                          corr_method, s, bl, basename(bl_csv)))
                next
            }
            b <- b %>% select(gene = Value, x = VAE, orig_baseline = ORIG)

            j <- inner_join(m, b, by = "gene")
            if (nrow(j) == 0) {
                notes <- c(notes, sprintf("[%s] %s / %s: no genes in common", corr_method, s, bl))
                next
            }
            # Both ORIG columns are corr(raw input, raw ground truth) and should agree;
            # a large gap means the two arms were not evaluated on the same matrices.
            orig_gap <- max(abs(j$orig_method - j$orig_baseline))
            if (orig_gap > 1e-6) {
                notes <- c(notes, sprintf("[%s] %s / %s: input-vs-ground-truth correlation differs between the two files by up to %.4g",
                                          corr_method, s, bl, orig_gap))
            }

            panel_rows[[length(panel_rows) + 1]] <- data.frame(
                corr_method = corr_method, sample = s, baseline = bl,
                x = j$x, y = j$y, stringsAsFactors = FALSE)

            stat_rows[[length(stat_rows) + 1]] <- data.frame(
                corr_method = corr_method, sample = s, baseline = bl,
                best_trans = bb$trans, best_norm = bb$norm,
                n_genes = nrow(j),
                mean_baseline = mean(j$x), mean_method = mean(j$y),
                mean_diff = mean(j$y - j$x),
                frac_method_higher = mean(j$y > j$x),
                orig_max_gap = orig_gap,
                stringsAsFactors = FALSE)
        }
        cat(sprintf("  %-34s best: %-14s / %-8s\n", s, bb$trans, bb$norm))
    }
}

if (length(panel_rows) == 0) stop("Nothing to plot: no (sample, baseline) pair had both files.")
dat   <- bind_rows(panel_rows)
stats <- bind_rows(stat_rows)

# ------------------------------------------------------------------ threshold sweep
# At threshold t keep only the genes whose correlation is >= t for BOTH arms, i.e. the
# quadrant up and to the right of (t, t) on the diagonal, then report what fraction of
# them sit above y = x. n_genes falls as t rises, so the curve is cut off once fewer than
# min_n genes remain rather than left to swing on a handful of points.
sweep_one <- function(x, y) {
    n_kept <- integer(length(THRESHOLDS))
    n_above <- integer(length(THRESHOLDS))
    for (i in seq_along(THRESHOLDS)) {
        sel <- x >= THRESHOLDS[i] & y >= THRESHOLDS[i]
        n_kept[i]  <- sum(sel)
        n_above[i] <- sum(y[sel] > x[sel])
    }
    data.frame(threshold = THRESHOLDS, n_kept = n_kept, n_above = n_above,
               pct_above = ifelse(n_kept > 0, 100 * n_above / n_kept, NA_real_),
               stringsAsFactors = FALSE)
}

curves <- dat %>%
    group_by(corr_method, sample, baseline) %>%
    group_modify(~ sweep_one(.x$x, .x$y)) %>%
    ungroup() %>%
    mutate(enough = n_kept >= min_n)

cat(sprintf("\nThreshold sweep: %d thresholds from %.2f to %.2f, curve cut below %d genes\n",
            length(THRESHOLDS), min(THRESHOLDS), max(THRESHOLDS), min_n))

# ------------------------------------------------------------------ plot
use_hexbin <- requireNamespace("hexbin", quietly = TRUE)

for (corr_method in corr_methods) {
    d <- dat   %>% filter(corr_method == !!corr_method)
    st <- stats %>% filter(corr_method == !!corr_method)
    if (nrow(d) == 0) {
        cat(sprintf("[skip] no panels for %s\n", corr_method))
        next
    }

    cv <- curves %>% filter(corr_method == !!corr_method)

    # Row order follows the sample table; the strip carries the selected trans / norm and,
    # when the sparsity table supplied one, the input modality's sparsity.
    row_label <- st %>%
        group_by(sample) %>%
        summarise(best_trans = first(best_trans), best_norm = first(best_norm), .groups = "drop") %>%
        mutate(sparsity = unname(sparsity_of[sample]),
               lab = sprintf("%s\n%s / %s%s", sample, best_trans, best_norm,
                             ifelse(is.na(sparsity), "", sprintf("\nsparsity: %.4f", sparsity))))
    lab_of <- setNames(row_label$lab, row_label$sample)
    ord <- samples[samples %in% names(lab_of)]

    d  <- d  %>% mutate(row = factor(unname(lab_of[sample]), levels = unname(lab_of[ord])),
                        baseline = factor(baseline, levels = BASELINES))
    st <- st %>% mutate(row = factor(unname(lab_of[sample]), levels = unname(lab_of[ord])),
                        baseline = factor(baseline, levels = BASELINES))
    cv <- cv %>% mutate(row = factor(unname(lab_of[sample]), levels = unname(lab_of[ord])),
                        baseline = factor(baseline, levels = BASELINES))

    p <- ggplot(d, aes(x = x, y = y))
    if (use_hexbin) p <- p + geom_hex(bins = 45) else p <- p + geom_bin_2d(bins = 45)

    p <- p +
        scale_fill_gradientn(colours = c("#440154", "#31688e", "#35b779", "#fde725"),
                             trans = "log10", name = "genes") +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                    colour = "red", linewidth = 0.4) +
        geom_text(data = st,
                  aes(x = -Inf, y = Inf,
                      label = sprintf("above y=x: %.1f%%\nmean diff: %+.3f",
                                      100 * frac_method_higher, mean_diff)),
                  inherit.aes = FALSE, hjust = -0.06, vjust = 1.12,
                  size = 2.4, colour = "grey20") +
        facet_grid(row ~ baseline, switch = "y") +
        coord_cartesian(xlim = c(AXIS_MIN, 1), ylim = c(AXIS_MIN, 1)) +
        scale_x_continuous(breaks = c(-0.2, 0, 0.5, 1)) +
        scale_y_continuous(breaks = c(-0.2, 0, 0.5, 1)) +
        labs(
            title = sprintf("%s  |  input modality: %s  |  %s correlation per gene",
                            folder_name, modality, corr_method),
            subtitle = sprintf("x: published baseline    y: %s at its best transformation and normalization per sample    above the dashed line favours %s",
                               method, method),
            x = "baseline correlation with ground truth",
            y = sprintf("%s correlation with ground truth", method)
        ) +
        theme_bw(base_size = 9) +
        theme(
            strip.text.y.left = element_text(angle = 0, size = 6.5, lineheight = 1.1),
            strip.text.x = element_text(size = 10, face = "bold"),
            strip.placement = "outside",
            panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold"),
            plot.subtitle = element_text(size = 7.5, colour = "grey30"),
            legend.position = "right"
        )

    n_rows <- length(levels(d$row))
    stem <- sprintf("%s_%s_%s_%s_baseline_vs_%s", folder_name, method, modality, corr_method, method)
    png_path <- file.path(out_dir, paste0(stem, ".png"))
    csv_path <- file.path(out_dir, paste0(stem, ".csv"))

    ggsave(png_path, p,
           width  = 3 * panel_in + 2.4,
           height = n_rows * panel_in + 1.6,
           dpi = 200, limitsize = FALSE)
    write.csv(st %>% select(-row), csv_path, row.names = FALSE)
    cat(sprintf("\n[SAVE] %s  (%d rows x 3)\n[SAVE] %s\n", png_path, n_rows, csv_path))

    # ---- threshold sweep figure, same grid ----
    cv_plot <- cv %>% filter(enough)
    if (nrow(cv_plot) == 0) {
        cat(sprintf("[skip] threshold sweep for %s: no panel keeps %d genes at any threshold\n",
                    corr_method, min_n))
    } else {
        q <- ggplot(cv_plot, aes(x = threshold, y = pct_above)) +
            geom_hline(yintercept = 50, linetype = "dotted", colour = "grey40", linewidth = 0.4) +
            geom_line(linewidth = 0.6, colour = "#31688e") +
            geom_point(size = 0.5, colour = "#31688e") +
            facet_grid(row ~ baseline, switch = "y") +
            coord_cartesian(xlim = c(AXIS_MIN, 1), ylim = c(0, 100)) +
            scale_x_continuous(breaks = c(-0.2, 0, 0.5, 1)) +
            labs(
                title = sprintf("%s  |  input modality: %s  |  %s correlation per gene",
                                folder_name, modality, corr_method),
                subtitle = sprintf("a gene is kept at threshold t only when its correlation is >= t for BOTH the baseline and %s\ny is the percentage of the kept genes above y = x, i.e. reconstructed closer to the ground truth by %s\ncurve truncated where fewer than %d genes remain; the dotted line at 50%% is parity",
                                   method, method, min_n),
                x = sprintf("correlation threshold t  (gene kept when baseline >= t and %s >= t)", method),
                y = sprintf("percentage of retained genes above y = x  (%s better)", method)
            ) +
            theme_bw(base_size = 9) +
            theme(
                strip.text.y.left = element_text(angle = 0, size = 6.5, lineheight = 1.1),
                strip.text.x = element_text(size = 10, face = "bold"),
                strip.placement = "outside",
                panel.grid.minor = element_blank(),
                plot.title = element_text(face = "bold"),
                plot.subtitle = element_text(size = 7.5, colour = "grey30"),
                legend.position = "none"
            )

        curve_png <- file.path(out_dir, paste0(stem, "_curve.png"))
        curve_csv <- file.path(out_dir, paste0(stem, "_curve.csv"))
        ggsave(curve_png, q,
               width  = 3 * panel_in + 1.2,
               height = n_rows * panel_in + 1.9,
               dpi = 200, limitsize = FALSE)
        write.csv(cv %>% select(-row), curve_csv, row.names = FALSE)
        cat(sprintf("[SAVE] %s  (%d rows x 3)\n[SAVE] %s\n", curve_png, n_rows, curve_csv))
    }
}

# ------------------------------------------------------------------ report
if (length(notes) > 0) {
    cat("\n=== Notes (panels skipped or worth checking) ===\n")
    cat(paste0("  ", notes, collapse = "\n"), "\n")
} else {
    cat("\nAll requested panels were produced.\n")
}