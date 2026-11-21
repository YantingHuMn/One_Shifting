# Combine existing raw-target prediction figures; no training or replotting.
# Usage: Rscript combine_prediction_tasks.R <base_dir> <out_dir> [sample_tsv]
suppressPackageStartupMessages({
    library(grid)
    library(png)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript combine_prediction_tasks.R <base_dir> <out_dir> [sample_tsv]")
base_dir <- args[1]
out_dir <- args[2]
if (length(args) >= 3 && nzchar(args[3])) {
    samples <- read.delim(args[3], stringsAsFactors = FALSE, check.names = FALSE)
    if (!all(c("folder_name", "sample_name") %in% names(samples))) {
        stop("sample_tsv must have folder_name and sample_name columns")
    }
    samples <- unique(samples[c("folder_name", "sample_name")])
} else {
    samples <- data.frame(
        folder_name = c("HCA", "HCA", "ENCODE", "ENCODE", "10x"),
        sample_name = c("HSB5871", "KTBpool12", "ENCSR056LAM", "ENCSR128ZLB",
                        "10k_PBMC_Multiome_nextgem_Chromium_Controller")
    )
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
manifest <- list()

# Top row: RNA input (0, 3, 5). Bottom row: ATAC input (1, 4, 2).
task_ids <- c(0L, 3L, 5L, 1L, 4L, 2L)
task_modalities <- c("RNA", "RNA", "RNA", "ATAC", "ATAC", "ATAC")

task_folder <- function(model, sample, task) {
    switch(as.character(task),
        "0" = sprintf("%s_within_%s_RNA_raw_target", model, sample),
        "1" = sprintf("%s_within_%s_ATAC_raw_target", model, sample),
        "2" = sprintf("%s_cross_%s_ATAC_to_%s_RNA_same_genes", model, sample, sample),
        "3" = sprintf("%s_cross_%s_RNA_to_%s_ATAC_TF_to_HVG", model, sample, sample),
        "4" = sprintf("%s_cross_%s_ATAC_to_%s_RNA_TF_to_HVG", model, sample, sample),
        "5" = sprintf("%s_cross_%s_RNA_to_%s_ATAC_same_genes", model, sample, sample)
    )
}

for (i in seq_len(nrow(samples))) {
    sample <- samples$sample_name[i]
    folder <- samples$folder_name[i]
    for (model in c("MLP", "Transformer")) {
        panels <- vector("list", 6)
        for (j in seq_along(task_ids)) {
            task <- task_ids[j]
            experiment <- task_folder(model, sample, task)
            fig_dir <- file.path(base_dir, folder, paste0(sample, "_INPUT_", task_modalities[j]),
                                 experiment, "figures")
            candidates <- list.files(fig_dir, pattern = "pearson_vs_spearman\\.png$", full.names = TRUE)
            # Restrict cross tasks to original scale and within tasks to raw counts.
            scale_tag <- if (task %in% c(0L, 1L)) "raw_counts" else "original_scale"
            candidates <- candidates[grepl(scale_tag, basename(candidates), fixed = TRUE)]
            path <- if (length(candidates) == 1) candidates[1] else NA_character_
            status <- if (length(candidates) == 0) "missing" else if (length(candidates) > 1) "ambiguous" else "ok"
            raster <- NULL
            if (status == "ok") {
                raster <- tryCatch(readPNG(path), error = function(e) {
                    warning(sprintf("Could not read %s: %s", path, conditionMessage(e)))
                    NULL
                })
                if (is.null(raster)) status <- "unreadable"
            }
            label <- if (status == "ok") basename(path) else paste0(experiment, " [", status, "]")
            panels[[j]] <- list(raster = raster, label = label, status = status, task = task)
            manifest[[length(manifest) + 1]] <- data.frame(
                folder_name = folder, sample_name = sample, model = model, task_id = task,
                figure_directory = fig_dir, source_png = path, status = status,
                stringsAsFactors = FALSE)
            if (status != "ok") warning(sprintf("%s / %s / task %s: %s (%s)", sample, model, task, status, fig_dir))
        }

        draw_figure <- function() {
            grid.newpage()
            grid.text(paste(sample, model, sep = " | "), x = .5, y = .98,
                      gp = gpar(fontsize = 16, fontface = "bold"))
            pushViewport(viewport(x = .5, y = .47, width = .98, height = .91,
                                  layout = grid.layout(2, 3)))
            for (j in seq_along(panels)) {
                panel <- panels[[j]]
                pushViewport(viewport(layout.pos.row = ceiling(j / 3), layout.pos.col = (j - 1) %% 3 + 1))
                starts <- seq.int(1L, nchar(panel$label), by = 64L)
                label <- paste(substring(panel$label, starts, pmin(starts + 63L, nchar(panel$label))), collapse = "\n")
                grid.text(paste0("Task ", panel$task, "\n", label), x = .5, y = .99,
                          just = "top", gp = gpar(fontsize = 8, lineheight = 1.05))
                pushViewport(viewport(x = .5, y = .43, width = .97, height = .80))
                if (!is.null(panel$raster)) {
                    dims <- dim(panel$raster)
                    # Fit without stretching; each cell has physical width 18*.98/3*.97
                    # and height 12*.91/2*.80 inches.
                    box_ratio <- (18 * .98 / 3 * .97) / (12 * .91 / 2 * .80)
                    ratio <- dims[2] / dims[1]
                    w <- if (ratio >= box_ratio) 1 else ratio / box_ratio
                    h <- if (ratio >= box_ratio) box_ratio / ratio else 1
                    grid.raster(panel$raster, width = unit(w, "npc"), height = unit(h, "npc"), interpolate = TRUE)
                } else {
                    grid.rect(gp = gpar(fill = "grey97", col = "grey80"))
                    grid.text(toupper(panel$status), gp = gpar(col = "grey40", fontsize = 13))
                }
                popViewport(2)
            }
            popViewport()
        }
        stem <- file.path(out_dir, paste0(sample, "_", model, "_six_tasks_raw_target"))
        grDevices::png(paste0(stem, ".png"), width = 18, height = 12, units = "in", res = 250, bg = "white")
        tryCatch(draw_figure(), finally = grDevices::dev.off())
        grDevices::pdf(paste0(stem, ".pdf"), width = 18, height = 12, bg = "white")
        tryCatch(draw_figure(), finally = grDevices::dev.off())
        cat("[SAVE]", paste0(stem, ".png"), "\n[SAVE]", paste0(stem, ".pdf"), "\n")
    }
}
write.csv(do.call(rbind, manifest), file.path(out_dir, "prediction_six_tasks_manifest.csv"), row.names = FALSE)
