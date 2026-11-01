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
#   within : space == "raw_counts"      (transformation AND normalization undone)
#   cross  : space == "original_scale"  (target is fixed no_norm + no_trans, so this is raw)
# Any other space, in particular "target_space" and the old "original_scale" of Within,
# is refused rather than silently drawn.
suppressPackageStartupMessages(
    library(ggplot2)
)

TRANS_LEVELS <- c("no_trans", "count+1", "sqrt", "sqrt+1", "log2", "log2(count+2)")
NORM_LEVELS  <- c("no_norm", "1000000", "100000", "10000", "1000")
N_EXPECTED   <- length(TRANS_LEVELS) * length(NORM_LEVELS)

# fixed, named maps: a given transformation / normalization looks the same in every figure
TRANS_COLORS <- setNames(c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628"),
                         TRANS_LEVELS)
NORM_SHAPES  <- setNames(c(16, 17, 15, 18, 4), NORM_LEVELS)

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
    if (!identical(space, ALLOWED_SPACE[[task]]))
        stop("task '", task, "' is drawn on space '", ALLOWED_SPACE[[task]], "' only; got --space '",
             space, "'. target_space and non-raw scales are deliberately not plotted.")
    check_label(sample, "sample"); check_label(modality, "modality"); check_label(space, "space")
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
    if (!(space %in% present_spaces)) {
        msg <- paste0("space '", space, "' not found in ", csv, ". Spaces present: ",
                      paste(present_spaces, collapse = ", "), ".")
        if (identical(space, "raw_counts"))
            msg <- paste0(msg, " Within needs raw_counts rows, which Step3_predict_MLP.py writes ",
                          "only when given --target_raw_matrix and --target_lib_sizes (ReLU / ",
                          "raw-scale version). Re-run Within into a new output directory; ",
                          "'original_scale' or 'target_space' rows are not used as a substitute.")
        stop(msg)
    }
    d <- df[df$space == space, , drop = FALSE]
    if (nrow(d) == 0) stop("No rows left after selecting space '", space, "'")

    bad_t <- setdiff(unique(d$input_trans), TRANS_LEVELS)
    bad_n <- setdiff(unique(d$input_norm), NORM_LEVELS)
    if (length(bad_t)) stop("Unsupported input_trans: ", paste(bad_t, collapse = ", "),
                            ". Supported: ", paste(TRANS_LEVELS, collapse = ", "))
    if (length(bad_n)) stop("Unsupported input_norm: ", paste(bad_n, collapse = ", "),
                            ". Supported: ", paste(NORM_LEVELS, collapse = ", "))

    key <- paste(d$input_trans, d$input_norm, sep = " | ")
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

    task_word <- if (task == "within") "Within-modality" else "Cross-modality"
    p <- ggplot(d, aes(x = x, y = y, color = input_trans, shape = input_norm)) +
        geom_point(size = 3.6, stroke = 1, alpha = 0.85) +
        scale_color_manual(values = TRANS_COLORS, limits = trans_present, name = "Input transformation") +
        scale_shape_manual(values = NORM_SHAPES, limits = norm_present, name = "Input normalization") +
        scale_x_continuous(expand = expansion(mult = 0.08)) +
        scale_y_continuous(expand = expansion(mult = 0.08)) +
        labs(
            x = "Mean gene-wise Pearson correlation",
            y = "Mean gene-wise Spearman correlation",
            title = sprintf("%s: %s MLP, %s", sample, task_word, pretty_modality(modality)),
            subtitle = sprintf("Evaluation scale: %s | %d/%d combinations plotted",
                               space, n_plot, N_EXPECTED),
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

    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    stem <- paste(sample, task, modality, space, "pearson_vs_spearman", sep = "_")
    png_path <- file.path(out_dir, paste0(stem, ".png"))
    pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
    ggsave(png_path, p, width = 8.5, height = 6, dpi = 300)
    ggsave(pdf_path, p, width = 8.5, height = 6, device = "pdf")
    cat(sprintf("Plotted %d/%d combinations\nSaved %s\nSaved %s\n", n_plot, N_EXPECTED,
                png_path, pdf_path))
    invisible(list(plot = p, data = d, png = png_path, pdf = pdf_path))
}

a <- parse_args(commandArgs(trailingOnly = TRUE))
run_mlp_summary_scatter_plot(a$csv, a$sample, a$task, a$modality, a$space, a$out_dir)