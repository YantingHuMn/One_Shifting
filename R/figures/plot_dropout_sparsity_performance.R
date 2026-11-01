# Dropout sparsity vs absolute reconstruction performance, across samples
#
# Post: For ATAC gene activity only, relates the sparsity actually measured on the
#   preprocessed input matrix to the absolute reconstruction performance of one method,
#   across five multinomial-downsampling keep fractions. One figure per correlation
#   metric: six transformation curves, each with at most five summary points.
#
#   Nothing here trains, reconstructs or preprocesses anything. It reads the summary CSVs
#   and the count matrices that the pipeline has already written, tolerates conditions
#   that have not finished, and can be re-run as more of them complete.
#
# Keep fractions and where each one lives, under
#   <base_dir>/<folder_name>/<sample_name>_INPUT_ATAC/ :
#   k = 0.3  dropout_0p3/      k = 0.5  dropout_0p5/
#   k = 0.7  dropout_0p7/      k = 0.9  dropout_0p9/
#   k = 1    the sample root itself -- no dropout_1 directory is expected or required.
#   The 0pN spelling is the driver's own tag (config_run_multi_ATAC_*.sh turns 0.3 into
#   0p3 before appending it to READ_DIR).
#
# Parameter:
#   base_dir     : One_Shifting_Results root, e.g. /dcs11/hongkai/data/yhu1/One_Shifting_Results
#   sample_tsv   : TSV with columns folder_name and sample_name, matched by NAME
#   method       : VAE, DCA_mse or Transformer_denoise -- only this method is required to
#                  be present; other methods in the same summary file are ignored
#   out_dir      : output directory
#   metrics      : comma separated, default "pearson,spearman"
#   summary_name : summary CSV file name, default
#                  bubble_plot_summary_col_gene_v1_raw__v2_raw__recon_revtrans_revnorm.csv
#                  It sits beside the method directories, not inside them.
# Output: see the closing report; PNG per metric, a sample-level long table, a
#   summary table, and an issues report.

suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
    library(arrow)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
    stop("Usage: Rscript plot_dropout_sparsity_performance.R <base_dir> <sample_tsv> ",
         "<method> <out_dir> [metrics] [summary_name] [performance_type]")
}
base_dir <- args[1]
sample_tsv <- args[2]
method <- args[3]
out_dir <- args[4]
metrics <- if (length(args) >= 5 && nzchar(args[5])) {
    trimws(strsplit(args[5], ",")[[1]])
} else {
    c("pearson", "spearman")
}
summary_name <- if (length(args) >= 6 && nzchar(args[6])) args[6] else
    "bubble_plot_summary_col_gene_v1_raw__v2_raw__recon_revtrans_revnorm.csv"

performance_type <- if (length(args) >= 7 && nzchar(args[7])) tolower(args[7]) else "absolute"
if (!performance_type %in% c("absolute", "relative")) {
    stop("performance_type must be absolute or relative, got: ", performance_type)
}
performance_column <- if (performance_type == "absolute") "mean_performance" else "mean_residual"

metrics <- tolower(metrics)
if (!all(metrics %in% c("pearson", "spearman"))) {
    stop("metrics must be pearson and/or spearman, got: ", paste(metrics, collapse = ", "))
}

# All six are kept and plotted; none is dropped and no best-of is taken across them.
TRANS_LEVELS <- c("no_trans", "count+1", "sqrt", "sqrt+1", "log2", "log2(count+2)")
TRANS_COLOURS <- c(
    "no_trans"      = "#999999",
    "count+1"       = "#E69F00",
    "sqrt"          = "#0072B2",
    "sqrt+1"        = "#56B4E9",
    "log2"          = "#009E73",
    "log2(count+2)" = "#D55E00"
)
# k = 1 maps to the sample root: no dropout_1 directory is looked for.
KEEPS <- c(0.3, 0.5, 0.7, 0.9, 1)
KEEP_SUBDIR <- c("0.3" = "dropout_0p3", "0.5" = "dropout_0p5",
                 "0.7" = "dropout_0p7", "0.9" = "dropout_0p9", "1" = "")
# Only the pipeline's own identifier column names are dropped by name; everything else is
# kept unless it is non-numeric. Speculative index-column names such as X1 are NOT listed,
# because a real feature column can legitimately carry that name and dropping it would
# change the sparsity denominator without saying so.
ID_COL_PATTERN <- "^(pos|gene|genes|gene_id|cell|cells|cell_id|index|rowname|rownames)$"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
issues <- character(0)
note <- function(fmt, ...) {
    msg <- sprintf(fmt, ...)
    issues <<- c(issues, msg)
    cat("  [note] ", msg, "\n", sep = "")
}

# ------------------------------------------------------------------ sample table
if (!file.exists(sample_tsv)) stop("Sample table not found: ", sample_tsv)
st <- read.delim(sample_tsv, stringsAsFactors = FALSE, check.names = FALSE)
if (!all(c("folder_name", "sample_name") %in% names(st))) {
    stop("Sample table must have columns folder_name and sample_name (has: ",
         paste(names(st), collapse = ", "), ")")
}
st <- st[nzchar(trimws(st$folder_name)) & nzchar(trimws(st$sample_name)), , drop = FALSE]
st$folder_name <- trimws(st$folder_name)
st$sample_name <- trimws(st$sample_name)
if (nrow(st) == 0) stop("No usable rows in ", sample_tsv)
cat(sprintf("Samples: %d from %s\n", nrow(st), basename(sample_tsv)))

cond_dir_of <- function(folder_name, sample_name, keep) {
    root <- file.path(base_dir, folder_name, paste0(sample_name, "_INPUT_ATAC"))
    sub <- KEEP_SUBDIR[[as.character(keep)]]
    if (nzchar(sub)) file.path(root, sub) else root
}

# sparsity
# Read the untransformed, unnormalized ATAC input once per (sample, keep); the resulting
# sparsity is shared by all six transformations and both metrics. Any NA / NaN / Inf in
# the numeric block makes the sparsity for that condition unusable: it is recorded and
# dropped rather than counted as zero or quietly removed from the denominator.
measure_sparsity <- function(cond_dir, sample_name, tag) {
    f <- file.path(cond_dir, paste0(sample_name, "_ATAC"), "Count_Matrix_norm_by_no_norm.feather")
    if (!file.exists(f)) {
        note("%s: ATAC count matrix missing (%s)", tag, f)
        return(list(sparsity = NA_real_, path = f, n_cells = NA_real_))
    }
    df <- tryCatch(read_feather(f), error = function(e) {
        note("%s: could not read %s (%s)", tag, basename(f), conditionMessage(e)); NULL })
    if (is.null(df)) return(list(sparsity = NA_real_, path = f, n_cells = NA_real_))

    drop_named <- grepl(ID_COL_PATTERN, tolower(names(df)))
    num_ok <- vapply(df, is.numeric, logical(1))
    keep_cols <- num_ok & !drop_named
    if (!any(keep_cols)) {
        note("%s: no numeric count columns in %s", tag, basename(f))
        return(list(sparsity = NA_real_, path = f, n_cells = NA_real_))
    }
    dropped <- names(df)[!keep_cols]
    if (length(dropped) > 0) {
        cat(sprintf("      %d count column(s) kept, %d excluded: %s\n",
                    sum(keep_cols), length(dropped),
                    paste(utils::head(dropped, 5), collapse = ", ")))
    }

    m <- as.matrix(df[, keep_cols, drop = FALSE])
    n_bad <- sum(!is.finite(m))
    if (n_bad > 0) {
        note("%s: %d non-finite value(s) (NA / NaN / Inf) in the count block of %s; sparsity skipped for this condition",
             tag, n_bad, basename(f))
        return(list(sparsity = NA_real_, path = f, n_cells = length(m)))
    }
    list(sparsity = sum(m == 0) / length(m), path = f, n_cells = length(m))
}

# gather
long_rows <- list()
sp_cache  <- list()

for (i in seq_len(nrow(st))) {
    fn <- st$folder_name[i]; sn <- st$sample_name[i]
    data_type_want <- paste0(sn, "_ATAC")           # must match in full
    cat(sprintf("\n%s / %s\n", fn, sn))

    for (k in KEEPS) {
        cond <- cond_dir_of(fn, sn, k)
        tag <- sprintf("%s/%s k=%s", fn, sn, k)
        if (!dir.exists(cond)) {
            note("%s: condition directory missing (%s)", tag, cond)
            next
        }

        sp <- measure_sparsity(cond, sn, tag)
        sp_cache[[tag]] <- sp
        cat(sprintf("    k=%-3s sparsity=%s\n", k,
                    if (is.na(sp$sparsity)) "unavailable" else sprintf("%.4f", sp$sparsity)))

        scsv <- file.path(cond, summary_name)
        if (!file.exists(scsv)) {
            note("%s: summary file missing (%s)", tag, scsv)
            next
        }
        bb <- tryCatch(read.csv(scsv, stringsAsFactors = FALSE, check.names = FALSE),
                       error = function(e) {
                           note("%s: could not read %s (%s)", tag, summary_name, conditionMessage(e)); NULL })
        if (is.null(bb)) next
        needed <- c("method", "data_type", "metric", "trans", "norm",
                    "mean_performance", "mean_residual")
        if (!all(needed %in% names(bb))) {
            note("%s: %s lacks %s (has: %s)", tag, summary_name,
                 paste(setdiff(needed, names(bb)), collapse = " / "),
                 paste(names(bb), collapse = ", "))
            next
        }

        sel <- bb[trimws(bb$method) == method &
                  trimws(bb$data_type) == data_type_want &
                  tolower(trimws(bb$metric)) %in% metrics, , drop = FALSE]
        if (nrow(sel) == 0) {
            note("%s: no rows for method == %s and data_type == %s in %s",
                 tag, method, data_type_want, summary_name)
            next
        }
        sel$trans  <- trimws(sel$trans)
        sel$metric <- tolower(trimws(sel$metric))

        off <- setdiff(sel$trans, TRANS_LEVELS)
        if (length(off) > 0) {
            note("%s: ignoring unrecognised transformation(s): %s", tag,
                 paste(unique(off), collapse = ", "))
            sel <- sel[sel$trans %in% TRANS_LEVELS, , drop = FALSE]
        }

        # The summary file is written with append = TRUE and no de-duplication, so the
        # same key can appear more than once. Report and drop every copy: averaging or
        # taking a maximum would silently invent a number the pipeline never produced.
        key <- paste(sel$metric, sel$trans, sep = "|")
        dup_keys <- unique(key[duplicated(key)])
        if (length(dup_keys) > 0) {
            for (dk in dup_keys) {
                vals <- sel$mean_performance[key == dk]
                note("%s: %d duplicate rows for metric|trans = %s (mean_performance %s); all skipped",
                     tag, length(vals), dk, paste(sprintf("%.6g", vals), collapse = ", "))
            }
            sel <- sel[!(key %in% dup_keys), , drop = FALSE]
        }
        if (nrow(sel) == 0) next

        long_rows[[length(long_rows) + 1]] <- data.frame(
            folder_name = fn, sample_name = sn, keep_fraction = k,
            sparsity = sp$sparsity, method = method,
            metric = sel$metric, trans = sel$trans, norm = as.character(sel$norm),
            mean_performance = as.numeric(sel$mean_performance),
            mean_residual = as.numeric(sel$mean_residual),
            summary_path = scsv, matrix_path = sp$path,
            stringsAsFactors = FALSE)
    }
}

long_path <- file.path(out_dir, sprintf("dropout_sparsity_performance_%s_sample_level.csv", method))
summ_path <- file.path(out_dir, sprintf("dropout_sparsity_performance_%s_summary.csv", method))
iss_path  <- file.path(out_dir, sprintf("dropout_sparsity_performance_%s_issues.txt", method))

write_issues <- function() {
    writeLines(c(sprintf("method: %s", method),
                 sprintf("summary file: %s", summary_name),
                 sprintf("samples in table: %d", nrow(st)),
                 "",
                 if (length(issues) == 0) "No missing, duplicate or invalid results were found."
                 else sprintf("%d item(s):", length(issues)),
                 if (length(issues) == 0) character(0) else paste0("  - ", issues)),
               iss_path)
}

# Finding nothing is itself a result worth reporting: the issues file says which
# directories, matrices, summary files or method rows were absent, so it is written
# before exiting rather than lost to an error.
if (length(long_rows) == 0) {
    write_issues()
    cat(sprintf("\nNo usable records were found under %s for method %s.\n[SAVE] %s\n",
                base_dir, method, iss_path))
    quit(save = "no", status = 0)
}
long <- bind_rows(long_rows)

for (m in metrics) {
    miss <- long %>% filter(metric == m) %>%
        group_by(folder_name, sample_name, keep_fraction) %>%
        summarise(have = n_distinct(trans), .groups = "drop") %>%
        filter(have < length(TRANS_LEVELS))
    if (nrow(miss) > 0) {
        for (r in seq_len(nrow(miss))) {
            got <- long %>% filter(metric == m, sample_name == miss$sample_name[r],
                                   keep_fraction == miss$keep_fraction[r]) %>% pull(trans)
            note("%s/%s k=%s %s: only %d of %d transformations present (missing: %s); excluded from this keep's common sample set",
                 miss$folder_name[r], miss$sample_name[r], miss$keep_fraction[r], m,
                 miss$have[r], length(TRANS_LEVELS),
                 paste(setdiff(TRANS_LEVELS, got), collapse = ", "))
        }
    }
}

# aggregate
# Per (metric, keep): the common sample set is the samples that have a valid result for
# ALL six transformations AND a valid sparsity. Samples are weighted equally -- genes are
# never pooled across samples and never used as weights.
summary_rows <- list()
counts_rows  <- list()

for (m in metrics) {
    for (k in KEEPS) {
        sub <- long %>% filter(metric == m, keep_fraction == k, !is.na(sparsity))
        common <- sub %>%
            group_by(folder_name, sample_name) %>%
            summarise(n_trans = n_distinct(trans), .groups = "drop") %>%
            filter(n_trans == length(TRANS_LEVELS))
        n_samp <- nrow(common)
        counts_rows[[length(counts_rows) + 1]] <-
            data.frame(metric = m, keep_fraction = k, n_samples = n_samp,
                       stringsAsFactors = FALSE)
        if (n_samp == 0) next

        sub <- sub %>% semi_join(common, by = c("folder_name", "sample_name"))
        # One sparsity per sample at this keep, so the x position is a plain unweighted
        # mean over samples and is identical for all six transformations.
        sp_per_sample <- sub %>%
            distinct(folder_name, sample_name, sparsity) %>% pull(sparsity)

        agg <- sub %>%
            group_by(trans) %>%
            summarise(mean_performance = mean(mean_performance),
                      sd_performance = if (n() > 1) sd(mean_performance) else NA_real_,
                      n_samples = n(), .groups = "drop") %>%
            mutate(metric = m, keep_fraction = k,
                   mean_sparsity = mean(sp_per_sample),
                   sd_sparsity = if (length(sp_per_sample) > 1) sd(sp_per_sample) else NA_real_)
        summary_rows[[length(summary_rows) + 1]] <- agg
    }
}

counts <- bind_rows(counts_rows)
if (length(summary_rows) == 0) {
    cat("\n[warn] no (metric, keep) cell had a complete common sample set; no figure drawn\n")
    summ <- data.frame()
} else {
    summ <- bind_rows(summary_rows) %>%
        mutate(trans = factor(trans, levels = TRANS_LEVELS)) %>%
        arrange(metric, trans, mean_sparsity)
}

# plot
for (m in metrics) {
    d <- long %>% filter(metric == m, is.finite(sparsity), is.finite(.data[[performance_column]])) %>%
        mutate(mean_sparsity = sparsity, trans = factor(trans, levels = TRANS_LEVELS),
               sample_id = factor(paste(folder_name, sample_name, sep = " / "),
                                  levels = unique(paste(st$folder_name, st$sample_name, sep = " / "))),
               k_label = factor(as.character(keep_fraction), levels = as.character(KEEPS)))
    cnt <- data.frame(keep_fraction = KEEPS, n_samples = vapply(KEEPS,
        function(k) n_distinct(d$sample_id[d$keep_fraction == k]), integer(1)))
    # n per keep comes from the valid data for THIS metric, never from the row count of
    # the sample table and never from the number of transformation records.
    sub_txt <- paste0("Samples used: ",
                      paste(sprintf("k=%s (n=%d)", cnt$keep_fraction, cnt$n_samples),
                            collapse = "; "))
    if (nrow(d) == 0) {
        cat(sprintf("\n[skip] %s: nothing to plot. %s\n", m, sub_txt))
        next
    }

    # All six transformations share an x at a given keep, because sparsity is a property
    # of the input matrix. One k label per x position therefore carries the same
    # information as six stacked copies and leaves the panel readable.
    klab <- d %>% group_by(keep_fraction) %>%
        summarise(mean_sparsity = first(mean_sparsity),
                  y = max(.data[[performance_column]], na.rm = TRUE), .groups = "drop")

    p <- ggplot(d, aes(x = mean_sparsity, y = .data[[performance_column]],
                       colour = trans, group = trans)) +
        geom_line(aes(linetype = trans), linewidth = 0.7, alpha = 0.85, na.rm = TRUE) +
        geom_point(aes(size = k_label), alpha = 0.85, na.rm = TRUE, show.legend = TRUE) +
        facet_wrap(~sample_id, ncol = 2) +
        scale_linetype_manual(values = c("dashed", "solid", "dashed", "solid", "dashed", "solid"),
                              breaks = TRANS_LEVELS, drop = FALSE, name = "transformation") +
        scale_size_manual(values = setNames(c(2.2, 2.8, 3.4, 4.0, 4.6), as.character(KEEPS)),
                          breaks = as.character(KEEPS), labels = paste0("k = ", KEEPS),
                          drop = FALSE, name = "Keep fraction") +
        scale_colour_manual(values = TRANS_COLOURS, breaks = TRANS_LEVELS,
                            drop = FALSE, name = "transformation") +
        scale_x_continuous(labels = function(v) sprintf("%.0f%%", 100 * v)) +
        expand_limits(y = max(d[[performance_column]], na.rm = TRUE) * 1.08) +
        labs(
            title = sprintf("%s  |  ATAC gene activity  |  %s", method, m),
            subtitle = sub_txt,
            x = "Input sparsity",
            y = if (performance_type == "absolute") "Mean gene-wise reconstruction correlation" else "Mean residual (input - reconstruction; lower is better)",
            caption = paste0(
                "Each panel shows one sample; points are not averaged across samples. Point size identifies the retained count fraction k.\n",
                "Lines connect observed points in increasing sparsity order; missing conditions are not imputed. No SD bars are plotted.\n",
                "Sparsity is measured on the preprocessed, untransformed, unnormalized ATAC input. Retained genes and cells can differ by k.\n",
                "Subtitle n counts samples with at least one valid transformation result at each keep fraction.\n",
                "Colors and line types identify transformations.")
        ) +
        theme_bw(base_size = 11) +
        theme(
            plot.title = element_text(face = "bold"),
            plot.subtitle = element_text(size = 9, colour = "grey25"),
            plot.caption = element_text(size = 6.6, colour = "grey35", hjust = 0),
            panel.grid.minor = element_blank(),
            legend.position = "right"
        )

    stem <- sprintf("dropout_sparsity_performance_%s_%s", method, m)
    if (performance_type == "relative") stem <- paste0(stem, "_", performance_type)
    ggsave(file.path(out_dir, paste0(stem, ".png")), p, width = 13.5,
           height = max(6, 4 * ceiling(n_distinct(d$sample_id) / 2) + 2),
           dpi = 300, limitsize = FALSE)
    cat("\n[SAVE] ", file.path(out_dir, paste0(stem, ".png")), "\n", sep = "")
}

# tables
write.csv(long %>% arrange(metric, trans, keep_fraction, folder_name, sample_name),
          long_path, row.names = FALSE)
if (nrow(summ) > 0) {
    write.csv(summ %>% select(metric, keep_fraction, trans, mean_sparsity, sd_sparsity,
                              mean_performance, sd_performance, n_samples),
              summ_path, row.names = FALSE)
}
write_issues()

cat(sprintf("[SAVE] %s\n[SAVE] %s\n[SAVE] %s\n",
            long_path, if (nrow(summ) > 0) summ_path else "(no summary rows)", iss_path))
cat(sprintf("\nIssues recorded: %d (see %s)\n", length(issues), basename(iss_path)))