# Summary scatter for the MLP tasks: one figure per sample x task, read straight from the
# compare_combinations_trans_norm.csv that Step3_predict_MLP{,_cross}.py maintains.
#
#   x = mean_pearson_defined   y = mean_spearman_defined
#   colour = input_trans       shape = input_norm       one point per trans x norm
#
# Both axes are the per-gene correlations already computed by the Python side (each target
# gene correlated across test cells, then averaged over the genes whose correlation is
# defined). Nothing is recomputed here and the input CSV is never modified.
#
# Evaluation scale per task:
#   within : raw_counts (both undone), target_space (normalization + transformation retained)
#   cross  : space == "original_scale"  (target is fixed no_norm + no_trans, so this is raw)
# Within accepts either scale or both as --space raw_counts,target_space.
# The old "original_scale" of Within is refused rather than silently drawn.
suppressPackageStartupMessages(
    library(ggplot2)
)

TRANS_LEVELS <- c("no_trans", "count+1", "sqrt", "sqrt+1", "log2", "log2(count+2)")
NORM_LEVELS  <- c("no_norm", "1000000", "100000", "10000", "1000")
N_EXPECTED   <- length(TRANS_LEVELS) * length(NORM_LEVELS)

# fixed, named maps: a given transformation / normalization looks the same in every figure
all_colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#F781BF")
all_shapes <- c(16, 17, 15, 18, 25, 21)
TRANS_COLORS <- setNames(all_colors[seq_along(TRANS_LEVELS)], TRANS_LEVELS)
NORM_SHAPES  <- setNames(all_shapes[seq_along(NORM_LEVELS)], NORM_LEVELS)

ALLOWED_SPACE <- list(within = "raw_counts", cross = "original_scale")
REQUIRED_COLS <- c("task", "space", "input_trans", "input_norm", "target_trans",
                   "target_norm", "mean_pearson_defined", "mean_spearman_defined")
X_COL <- "mean_pearson_defined"
Y_COL <- "mean_spearman_defined"

parse_args <- function(argv) {
    keys <- c("csv", "sample", "task", "modality", "space", "out_dir")
    out <- list()
    i <- 1
    while (i <= length(argv)) {
        a <- argv[i]
        if (!grepl("^--", a)) stop("Unexpected argument '", a, "'; expected --name value pairs")
        k <- sub("^--", "", a)
        if (!(k %in% keys)) stop("Unknown argument --", k, "; supported: ",
                                 paste0("--", keys, collapse = ", "))
        if (i == length(argv)) stop("Argument --", k, " has no value")
        out[[k]] <- argv[i + 1]
        i <- i + 2
    }
    missing <- setdiff(keys, names(out))
    if (length(missing)) stop("Missing required argument(s): ",
                              paste0("--", missing, collapse = ", "))
    for (k in keys) if (!nzchar(out[[k]])) stop("Argument --", k, " is empty")
    out
}

# labels end up in file names, so keep them to a safe character set
check_label <- function(value, name) {
    if (!grepl("^[A-Za-z0-9._+-]+$", value))
        stop("--", name, " '", value, "' contains characters not allowed in a file name ",
             "(allowed: letters, digits, . _ + -)")
}

pretty_modality <- function(m) gsub("ATAC", "ATAC gene activity", gsub("_to_", " to ", m))

run_mlp_summary_scatter_plot <- function(csv, sample, task, modality, space, out_dir) {
    if (!(task %in% names(ALLOWED_SPACE)))
        stop("--task must be 'within' or 'cross', got '", task, "'")
    spaces <- strsplit(space, ",", fixed = TRUE)[[1]]
    allowed <- if (task == "within") c("raw_counts", "target_space") else ALLOWED_SPACE[[task]]
    if (anyDuplicated(spaces) || !length(spaces) || any(!spaces %in% allowed))
        stop("Unsupported or duplicated --space for task '", task, "': ", space)
    check_label(sample, "sample"); check_label(modality, "modality")
    for (sp in spaces) check_label(sp, "space")
    if (!file.exists(csv)) stop("CSV not found: ", csv)

    # everything read as character: normalization labels such as 1000000 must stay labels
    df <- read.csv(csv, colClasses = "character", stringsAsFactors = FALSE,
                   check.names = FALSE)
    cat(sprintf("Read %d row(s) from %s\n", nrow(df), csv))
    absent <- setdiff(REQUIRED_COLS, colnames(df))
    if (length(absent)) stop("CSV is missing required column(s): ", paste(absent, collapse = ", "))

    other_tasks <- setdiff(unique(df$task), task)
    if (length(other_tasks))
        stop("CSV contains task(s) other than '", task, "': ", paste(other_tasks, collapse = ", "),
             ". One figure must come from one task's CSV.")

    present_spaces <- unique(df$space)
    missing_spaces <- setdiff(spaces, present_spaces)
    if (length(missing_spaces))
        stop("Requested space(s) missing: ", paste(missing_spaces, collapse = ", "))
    d <- df[df$space %in% spaces, , drop = FALSE]
    if (nrow(d) == 0) stop("No rows left after selecting spaces '", space, "'")

    bad_t <- setdiff(unique(d$input_trans), TRANS_LEVELS)
    bad_n <- setdiff(unique(d$input_norm), NORM_LEVELS)
    if (length(bad_t)) stop("Unsupported input_trans: ", paste(bad_t, collapse = ", "),
                            ". Supported: ", paste(TRANS_LEVELS, collapse = ", "))
    if (length(bad_n)) stop("Unsupported input_norm: ", paste(bad_n, collapse = ", "),
                            ". Supported: ", paste(NORM_LEVELS, collapse = ", "))

    key <- paste(d$space, d$input_trans, d$input_norm, sep = " | ")
    dup <- unique(key[duplicated(key)])
    if (length(dup))
        stop("Duplicate rows for the same input_trans | input_norm in space '", space, "': ",
             paste(dup, collapse = "; "),
             ". Refusing to average or pick one; clean the CSV or use a fresh output directory.")

    if (task == "cross") {
        if (!all(d$target_norm == "no_norm") || !all(d$target_trans == "no_trans"))
            stop("Cross is defined with a fixed target (no_norm + no_trans), but the selected rows ",
                 "have target_norm = {", paste(unique(d$target_norm), collapse = ", "),
                 "} and target_trans = {", paste(unique(d$target_trans), collapse = ", "), "}")
    }

    d$x <- suppressWarnings(as.numeric(d[[X_COL]]))
    d$y <- suppressWarnings(as.numeric(d[[Y_COL]]))
    finite <- is.finite(d$x) & is.finite(d$y)
    if (any(!finite)) {
        cat(sprintf("[WARN] %d combination(s) have a non-finite coordinate and are skipped:\n",
                    sum(!finite)))
        for (i in which(!finite))
            cat(sprintf("  input_trans=%s input_norm=%s  %s=%s  %s=%s\n", d$input_trans[i],
                        d$input_norm[i], X_COL, d[[X_COL]][i], Y_COL, d[[Y_COL]][i]))
    }
    d <- d[finite, , drop = FALSE]
    if (nrow(d) == 0) stop("No combination has finite ", X_COL, " and ", Y_COL,
                           " in space '", space, "'; nothing to plot")

    d$input_trans <- factor(d$input_trans, levels = TRANS_LEVELS)
    d$input_norm  <- factor(d$input_norm,  levels = NORM_LEVELS)
    trans_present <- TRANS_LEVELS[TRANS_LEVELS %in% as.character(d$input_trans)]
    norm_present  <- NORM_LEVELS[NORM_LEVELS %in% as.character(d$input_norm)]
    n_plot <- nrow(d)
    panel_counts <- vapply(spaces, function(sp) sum(d$space == sp), integer(1))
    if (any(panel_counts == 0)) stop("No finite points in space(s): ",
                                   paste(spaces[panel_counts == 0], collapse = ", "))
    d$evaluation_space <- factor(d$space, levels = spaces, labels = paste0(spaces, " (", panel_counts, "/", N_EXPECTED, ")"))

    lists_dir <- file.path(dirname(csv), "gene_lists")
    input_genes <- readLines(file.path(lists_dir, "input_genes.txt"), warn = FALSE)
    target_genes <- readLines(file.path(lists_dir, "target_genes.txt"), warn = FALSE)
    n_input <- sum(nzchar(trimws(input_genes)))
    n_target <- sum(nzchar(trimws(target_genes)))
    n_test <- unique(d$n_test_cells)
    if (length(n_test) != 1) stop("Inconsistent n_test_cells across selected combinations")

    task_word <- if (task == "within") "Within-modality" else "Cross-modality"
    p <- ggplot(d, aes(x = x, y = y, color = input_trans, shape = input_norm)) +
        geom_abline(slope = 1, intercept = 0, color = "grey35", linewidth = 0.7, linetype = "dashed") +
        geom_point(size = 3.6, stroke = 1, alpha = 0.85) +
        scale_color_manual(values = TRANS_COLORS, limits = trans_present, name = "Input transformation") +
        scale_shape_manual(values = NORM_SHAPES, limits = norm_present, name = "Input normalization") +
        scale_x_continuous(expand = expansion(mult = 0.08)) +
        scale_y_continuous(expand = expansion(mult = 0.08)) +
        labs(
            x = "Mean gene-wise Pearson correlation",
            y = "Mean gene-wise Spearman correlation",
            title = sprintf("%s: %s MLP, %s", sample, task_word, pretty_modality(modality)),
            subtitle = sprintf(
                "Input: %d genes | Target: %d genes | Test: %s cells\nEvaluation scales: %s",
                n_input, n_target, n_test, paste(spaces, collapse = " | ")
            ),
            caption = "Mean correlations over genes with defined correlations."
        ) +
        theme_bw() +
        theme(
            legend.position = "right",
            plot.title = element_text(size = 11, hjust = 0.5),
            plot.subtitle = element_text(size = 9, hjust = 0.5),
            plot.caption = element_text(size = 8, hjust = 0),
            legend.text = element_text(size = 9),
            legend.title = element_text(size = 10, face = "bold")
        )

    if (length(spaces) > 1)
        p <- p + facet_wrap(~evaluation_space, nrow = 1, scales = "fixed")

    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    stem <- paste(sample, task, modality, paste(spaces, collapse = "_and_"), "pearson_vs_spearman", sep = "_")
    png_path <- file.path(out_dir, paste0(stem, ".png"))
    pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
    ggsave(png_path, p, width = if (length(spaces) > 1) 14 else 8.5, height = 6, dpi = 300)
    ggsave(pdf_path, p, width = if (length(spaces) > 1) 14 else 8.5, height = 6, device = "pdf")
    cat(sprintf("Plotted %d points across %d scale(s)\nSaved %s\nSaved %s\n", n_plot, length(spaces),
                png_path, pdf_path))
    invisible(list(plot = p, data = d, png = png_path, pdf = pdf_path))
}

a <- parse_args(commandArgs(trailingOnly = TRUE))
run_mlp_summary_scatter_plot(a$csv, a$sample, a$task, a$modality, a$space, a$out_dir)