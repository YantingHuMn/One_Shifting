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
#   folder_name  : HCA, ENCODE or 10x; a comma separated list such as HCA,ENCODE,10x is only the
#                  set of folders searched for each sample's result directory
#   method       : the MSE arm to compare against, e.g. VAE, DCA_mse, Transformer_denoise
#   modality     : RNA or ATAC -- which modality is the model input
#   out_dir      : where the figures are written, e.g. <base_dir>/Paper_Figures
#   corr_methods : comma separated, default "pearson,spearman"
#   tsv_path     : sample table override; defaults to <base_dir>/<folder_name>_input_sample_pairs.tsv
#                  (one merged table is fine, and so is a comma separated list of per-folder tables,
#                  which are read in order: only the sample_name column is read, else column 2,
#                  else the only column)
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
# the latter case the entry whose file name starts with folder_name is used, unless
# folder_name is itself a comma separated list, when every listed table is read in order.
tsv_path <- if (length(args) >= 7 && nzchar(args[7])) {
    cand <- trimws(strsplit(args[7], ",")[[1]])
    cand <- cand[nzchar(cand)]
    if (length(cand) == 1 || grepl(",", folder_name)) {
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
samples <- character(0)
for (tsv_file in tsv_path) {
    if (!file.exists(tsv_file)) stop("Sample table not found: ", tsv_file)
    tsv <- read.delim(tsv_file, stringsAsFactors = FALSE)
    if (ncol(tsv) < 1) stop("Sample table has no columns: ", tsv_file)
    sample_col <- if ("sample_name" %in% names(tsv)) "sample_name" else if (ncol(tsv) >= 2) 2 else 1
    samples <- c(samples, trimws(as.character(tsv[[sample_col]])))
}
samples <- unique(samples[!is.na(samples) & nzchar(samples)])
if (length(samples) == 0) stop("No sample names in ", paste(tsv_path, collapse = ", "))
cat(sprintf("Samples from %s: %d\n", paste(basename(tsv_path), collapse = ", "), length(samples)))

other_modality <- if (modality == "ATAC") "RNA" else "ATAC"

# ------------------------------------------------------------------ locate each sample
# folder_name may be a comma separated search scope. A sample lives in the folder whose
# <folder>/<sample>_INPUT_<modality> directory actually exists; the sample table is never
# used to infer the folder. A sample found in no folder, or in more than one, is reported
# and left out rather than guessed.
folder_scope <- trimws(strsplit(folder_name, ",")[[1]])
folder_scope <- unique(folder_scope[nzchar(folder_scope)])
folder_of <- setNames(rep(NA_character_, length(samples)), samples)
loc_notes <- character(0)
for (s in samples) {
    cand_dirs <- file.path(base_dir, folder_scope, paste0(s, "_INPUT_", modality))
    found <- dir.exists(cand_dirs)
    if (sum(found) == 1) {
        folder_of[[s]] <- folder_scope[found]
    } else if (sum(found) == 0) {
        loc_notes <- c(loc_notes, sprintf("%s: sample directory not found; searched: %s",
                                          s, paste(cand_dirs, collapse = " | ")))
    } else {
        loc_notes <- c(loc_notes, sprintf("%s: ambiguous, the sample directory exists in more than one folder (%s); not used: %s",
                                          s, paste(folder_scope[found], collapse = ", "),
                                          paste(cand_dirs[found], collapse = " | ")))
    }
}
for (msg in loc_notes) cat(sprintf("[warn] %s\n", msg))
samples <- samples[!is.na(folder_of[samples])]
if (length(samples) == 0) stop("No sample directory was found for any sample under: ", paste(folder_scope, collapse = ", "))
cat(sprintf("Samples located: %d\n", length(samples)))

# ------------------------------------------------------------------ sparsity (optional)
# <folder>_sparsity.csv carries one row per (dataset, sample_name, data_type). The row for
# a sample is found by matching those three key columns by name -- never by row position,
# and never by assuming the input modality comes first. An absent file, an absent sample,
# or an unusable table costs the label only; the figures are still produced.
sparsity_of <- setNames(numeric(0), character(0))
folder_name_all <- folder_name
for (folder_name in unique(folder_of[samples])) {
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
                sparsity_of <- c(sparsity_of, setNames(as.numeric(sp$sparsity[hit]), sn[hit]))
                cat(sprintf("Sparsity table: %d rows matched dataset == %s and data_type == %s in %s\n",
                            length(hit), folder_name, modality, basename(sparsity_csv)))
                missing_sp <- setdiff(samples[folder_of[samples] == folder_name], names(sparsity_of))
                if (length(missing_sp) > 0) {
                    cat(sprintf("[warn] no %s sparsity row for: %s\n",
                                modality, paste(missing_sp, collapse = ", ")))
                }
            }
        }
    }
}
}
folder_name <- folder_name_all

sample_dir_of <- function(s) file.path(base_dir, folder_of[[s]], paste0(s, "_INPUT_", modality))
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
notes <- loc_notes
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

    column_stats <- st %>%
        filter(n_genes > 0, is.finite(frac_method_higher)) %>%
        group_by(baseline) %>%
        summarise(
            n_above = sum(frac_method_higher > 0.5),
            n_valid = n(),
            .groups = "drop"
        )

    baseline_labels <- setNames(
        sprintf(
            "%s\n%d/%d samples\n>50%% above y=x",
            column_stats$baseline,
            column_stats$n_above,
            column_stats$n_valid
        ),
        as.character(column_stats$baseline)
    )

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
        facet_grid(
            row ~ baseline,
            switch = "y",
            labeller = labeller(baseline = as_labeller(baseline_labels))
        ) +
        coord_cartesian(xlim = c(AXIS_MIN, 1), ylim = c(AXIS_MIN, 1)) +
        scale_x_continuous(breaks = c(-0.25, 0, 0.5, 1)) +
        scale_y_continuous(breaks = c(-0.25, 0, 0.5, 1)) +
        labs(
            title = sprintf("%s  |  input modality: %s  |  %s correlation per gene",
                            folder_name, modality, corr_method),
            subtitle = sprintf("x: published baseline    y: %s at its best transformation and normalization per sample  \n  above the dashed line favours %s",
                               method, method),
            x = "baseline correlation with ground truth",
            y = sprintf("%s correlation with ground truth", method)
        ) +
        theme_bw(base_size = 9) +
        theme(
            strip.text.y.left = element_text(angle = 0, size = 6.5, lineheight = 1.1),
            strip.text.x = element_text(size = 8, face = "bold", lineheight = 1.05),
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
            scale_x_continuous(breaks = c(-0.25, 0, 0.5, 1)) +
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

    # ---- one square 6 x 6 grid per baseline, same data and styling as the long scatter ----
    # Every panel is one sample, in the row order of the long scatter. Fewer than 36
    # samples leaves the remaining cells blank; more than 36 is split into pages in the
    # same order, so no sample is dropped.
    PANELS_PER_PAGE <- 36L
    for (bl in BASELINES) {
        d_bl  <- d  %>% filter(baseline == bl)
        st_bl <- st %>% filter(baseline == bl)
        if (nrow(d_bl) == 0) {
            cat(sprintf("[skip] 6x6 grid for %s / %s: no panels\n", corr_method, bl))
            next
        }
        ord_bl <- ord[ord %in% unique(st_bl$sample)]
        cs <- column_stats %>% filter(baseline == bl)
        n_above_bl <- if (nrow(cs) == 1) cs$n_above[1] else 0L
        n_valid_bl <- if (nrow(cs) == 1) cs$n_valid[1] else 0L
        pages <- split(ord_bl, ceiling(seq_along(ord_bl) / PANELS_PER_PAGE))

        for (pg in seq_along(pages)) {
            keep <- pages[[pg]]
            # facet_wrap ignores nrow once the panels fit in fewer rows, so the unused
            # cells of the 6 x 6 page are held open by empty factor levels. They are
            # distinct blank strings, which keeps them unique and leaves the strip blank.
            lev  <- unname(lab_of[keep])
            if (length(lev) < PANELS_PER_PAGE) {
                lev <- c(lev, vapply(seq_len(PANELS_PER_PAGE - length(lev)),
                                     function(k) strrep(" ", k), character(1)))
            }
            dp  <- d_bl  %>% filter(sample %in% keep) %>%
                mutate(row = factor(unname(lab_of[sample]), levels = lev))
            stp <- st_bl %>% filter(sample %in% keep) %>%
                mutate(row = factor(unname(lab_of[sample]), levels = lev))

            g <- ggplot(dp, aes(x = x, y = y))
            if (use_hexbin) g <- g + geom_hex(bins = 45) else g <- g + geom_bin_2d(bins = 45)

            # the diagonal is drawn only in the occupied cells, so a padding cell stays
            # empty instead of looking like a sample whose panel happens to have no points
            ab <- data.frame(row = factor(unname(lab_of[keep]), levels = lev),
                             slope = 1, intercept = 0)

            g <- g +
                scale_fill_gradientn(colours = c("#440154", "#31688e", "#35b779", "#fde725"),
                                     trans = "log10", name = "genes") +
                geom_abline(data = ab, aes(slope = slope, intercept = intercept),
                            inherit.aes = FALSE, linetype = "dashed",
                            colour = "red", linewidth = 0.4) +
                geom_text(data = stp,
                          aes(x = -Inf, y = Inf,
                              label = sprintf("above y=x: %.1f%%\nmean diff: %+.3f",
                                              100 * frac_method_higher, mean_diff)),
                          inherit.aes = FALSE, hjust = -0.06, vjust = 1.12,
                          size = 2.4, colour = "grey20") +
                facet_wrap(~ row, ncol = 6, drop = FALSE) +
                coord_cartesian(xlim = c(AXIS_MIN, 1), ylim = c(AXIS_MIN, 1)) +
                scale_x_continuous(breaks = c(-0.25, 0, 0.5, 1)) +
                scale_y_continuous(breaks = c(-0.25, 0, 0.5, 1)) +
                labs(
                    title = sprintf("%s  |  input modality: %s  |  baseline: %s  |  %s correlation per gene%s",
                                    folder_name, modality, bl, corr_method,
                                    if (length(pages) > 1) sprintf("  |  page %d of %d", pg, length(pages)) else ""),
                    subtitle = sprintf("%d/%d samples >50%% above y=x    x: %s    y: %s at its best transformation and normalization per sample",
                                       n_above_bl, n_valid_bl, bl, method),
                    x = "baseline correlation with ground truth",
                    y = sprintf("%s correlation with ground truth", method)
                ) +
                theme_bw(base_size = 9) +
                theme(
                    aspect.ratio = 1,
                    strip.text = element_text(size = 6, lineheight = 1.1),
                    # six columns leave little room per panel, so the shared tick labels
                    # are shrunk rather than the breaks changed
                    axis.text = element_text(size = 5.5),
                    panel.grid.minor = element_blank(),
                    plot.title = element_text(face = "bold"),
                    plot.subtitle = element_text(size = 7.5, colour = "grey30"),
                    legend.position = "right"
                )

            sq_png <- file.path(out_dir, sprintf("%s_%s_square_6x6%s.png", stem, bl,
                                                 if (length(pages) > 1) sprintf("_page%d", pg) else ""))
            ggsave(sq_png, g,
                   width  = 6 * panel_in + 1.8,
                   height = 6 * panel_in + 2.2,
                   dpi = 200, limitsize = FALSE)
            cat(sprintf("[SAVE] %s  (%d panels)\n", sq_png, length(keep)))
        }
    }
}

# ------------------------------------------------------------------ sample wins bar
# One figure, six bars: each correlation x baseline, counting SAMPLES rather than genes.
# Built from the same `stats` rows and the same validity filter as the scatter column
# titles, so the two always agree; the threshold-sweep curves are not used here. A sample
# with no usable result for a baseline is absent from `stats` and is therefore not counted
# in either colour rather than counted as a loss.
win_rows <- stats %>%
    filter(n_genes > 0, is.finite(frac_method_higher))

if (nrow(win_rows) == 0) {
    cat("\n[skip] sample wins bar: no (sample, baseline) pair had a usable result\n")
} else {
    OUT_LEVELS <- c(sprintf("%s above y=x in >50%% of genes", method),
                    sprintf("%s at or below 50%%", method))
    win_stats <- win_rows %>%
        mutate(outcome = ifelse(frac_method_higher > 0.5, OUT_LEVELS[1], OUT_LEVELS[2])) %>%
        count(corr_method, baseline, outcome, name = "n_samples") %>%
        mutate(corr_method = factor(corr_method, levels = corr_methods),
               baseline    = factor(baseline, levels = BASELINES),
               outcome     = factor(outcome, levels = OUT_LEVELS))
    bar_totals <- win_stats %>%
        group_by(corr_method, baseline) %>%
        summarise(n_valid = sum(n_samples), .groups = "drop")

    # facet_wrap leaves scales fixed, so both correlations share one sample-count axis.
    bar <- ggplot(win_stats, aes(x = baseline, y = n_samples, fill = outcome)) +
        geom_col(width = 0.68) +
        geom_text(aes(label = n_samples), position = position_stack(vjust = 0.5),
                  size = 3, colour = "white", fontface = "bold") +
        geom_text(data = bar_totals, aes(x = baseline, y = n_valid, label = n_valid),
                  inherit.aes = FALSE, vjust = -0.6, size = 3, colour = "grey20") +
        scale_fill_manual(values = setNames(c("#8AA6CE", "#E3AEBE"), OUT_LEVELS), name = NULL) +
        scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
        facet_wrap(~ corr_method, nrow = 1) +
        labs(
            title = sprintf("%s  |  input modality: %s  |  samples where %s beats each baseline",
                            folder_name, modality, method),
            subtitle = sprintf("one bar per baseline and correlation; a sample counts as a win when %s is above y = x for more than half of its genes\nbar height is the number of samples with a usable result for that baseline, printed above each bar",
                               method),
            x = NULL,
            y = "samples"
        ) +
        theme_bw(base_size = 9) +
        theme(
            strip.text = element_text(size = 10, face = "bold"),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank(),
            plot.title = element_text(face = "bold"),
            plot.subtitle = element_text(size = 7.5, colour = "grey30"),
            legend.position = "bottom"
        )

    bar_stem <- sprintf("%s_%s_%s_sample_wins_bar", folder_name, method, modality)
    bar_png  <- file.path(out_dir, paste0(bar_stem, ".png"))
    bar_csv  <- file.path(out_dir, paste0(bar_stem, ".csv"))
    ggsave(bar_png, bar, width = 7.6, height = 4.6, dpi = 200, limitsize = FALSE)
    write.csv(win_stats %>%
                  left_join(bar_totals, by = c("corr_method", "baseline")) %>%
                  arrange(corr_method, baseline, outcome),
              bar_csv, row.names = FALSE)
    cat(sprintf("\n[SAVE] %s\n[SAVE] %s\n", bar_png, bar_csv))
}

# ------------------------------------------------------------------ report
if (length(notes) > 0) {
    cat("\n=== Notes (panels skipped or worth checking) ===\n")
    cat(paste0("  ", notes, collapse = "\n"), "\n")
} else {
    cat("\nAll requested panels were produced.\n")
}