summarize_peak_to_gene_activity <- function(input_path, output_recon_path, INPUT_TRANS_PATH, INPUT_TRANS_PATH_filtered, GROUND_TRUTH_path, filtered_GROUND_TRUTH_path, upstream = 2000) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(arrow)
        library(Matrix)
        library(GenomicRanges)
        library(GenomeInfoDb)
        library(Signac)
        library(EnsDb.Hsapiens.v86)
        library(dplyr)
    })

    summarize_one_peak_matrix <- function(peak_input_path, peak_gr, gene_coords, overlaps) {
        cat("Reading peak input:", peak_input_path, "\n")
        df <- read_feather(peak_input_path)
        df <- as.data.frame(df)

        if (ncol(df) < 2) {
            stop("Input file must contain one cell/barcode column and peak columns.")
        }

        cell_col <- colnames(df)[1]
        cell_names <- df[[cell_col]]

        peak_names <- colnames(df)[-1]

        cat("Input cells:", nrow(df), "\n")
        cat("Input peaks:", length(peak_names), "\n")

        if (!identical(peak_names, names(peak_gr))) {
            stop("Peak columns in this file do not match the original input_path peak columns.")
        }

        peak_mat <- as.matrix(df[, -1, drop = FALSE])
        storage.mode(peak_mat) <- "numeric"

        peak_mat <- t(peak_mat)
        rownames(peak_mat) <- peak_names
        colnames(peak_mat) <- cell_names
        peak_mat <- Matrix(peak_mat, sparse = TRUE)

        gene_names <- gene_coords$gene_name

        gene_activity <- matrix(
            0,
            nrow = length(gene_coords),
            ncol = ncol(peak_mat),
            dimnames = list(gene_names, colnames(peak_mat))
        )

        for (i in seq_along(gene_coords)) {
            peak_idx <- queryHits(overlaps)[subjectHits(overlaps) == i]

            if (length(peak_idx) > 0) {
                if (length(peak_idx) == 1) {
                    gene_activity[i, ] <- peak_mat[peak_idx, ]
                } else {
                    gene_activity[i, ] <- Matrix::colSums(peak_mat[peak_idx, , drop = FALSE])
                }
            }
        }

        keep_genes <- rowSums(gene_activity) > 0
        gene_activity <- gene_activity[keep_genes, , drop = FALSE]

        cat("Genes with nonzero activity:", nrow(gene_activity), "\n")

        gene_activity <- t(gene_activity)

        gene_activity_df <- as.data.frame(gene_activity)
        gene_activity_df <- cbind(pos = rownames(gene_activity_df), gene_activity_df)
        rownames(gene_activity_df) <- NULL

        return(gene_activity_df)
    }

    # Check if reconstruction file exists
    if (!file.exists(input_path)) {
        warning("Reconstruction file not found: ", input_path, ". Skipping summarization.")
        return(invisible(NULL))
    }

    cat("Reading input:", input_path, "\n")
    df <- read_feather(input_path)
    df <- as.data.frame(df)

    if (ncol(df) < 2) {
        stop("Input file must contain one cell/barcode column and peak columns.")
    }

    peak_names <- colnames(df)[-1]

    cat("Input cells:", nrow(df), "\n")
    cat("Input peaks:", length(peak_names), "\n")

    peak_df <- do.call(rbind, strsplit(peak_names, "[:-]"))

    if (ncol(peak_df) < 3) {
        stop("Peak names must look like chr1-100-200 or chr1:100-200.")
    }

    peak_gr <- GRanges(
        seqnames = peak_df[, 1],
        ranges = IRanges(
            start = as.numeric(peak_df[, 2]),
            end = as.numeric(peak_df[, 3])
        )
    )
    names(peak_gr) <- peak_names

    annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)

    old_seq <- seqlevels(annotations)
    new_seq <- ifelse(old_seq == "MT", "chrM", paste0("chr", old_seq))
    names(new_seq) <- old_seq
    annotations <- renameSeqlevels(annotations, new_seq)
    genome(annotations) <- "hg38"

    gene_df <- data.frame(
        chr = as.character(seqnames(annotations)),
        start = start(annotations),
        end = end(annotations),
        strand = as.character(strand(annotations)),
        gene_name = annotations$gene_name
    )

    gene_summary <- gene_df %>%
        filter(!is.na(gene_name), gene_name != "") %>%
        group_by(gene_name) %>%
        summarise(
            chr = names(sort(table(chr), decreasing = TRUE))[1],
            start = min(start),
            end = max(end),
            strand = names(sort(table(strand), decreasing = TRUE))[1],
            .groups = "drop"
        )

    gene_coords <- GRanges(
        seqnames = gene_summary$chr,
        ranges = IRanges(
            start = ifelse(
                gene_summary$strand == "+",
                gene_summary$start - upstream,
                gene_summary$start
            ),
            end = ifelse(
                gene_summary$strand == "-",
                gene_summary$end + upstream,
                gene_summary$end
            )
        ),
        strand = gene_summary$strand,
        gene_name = gene_summary$gene_name
    )

    start(gene_coords) <- pmax(start(gene_coords), 1)
    gene_coords <- keepStandardChromosomes(gene_coords, pruning.mode = "coarse")

    cat("Number of genes in annotation:", length(gene_coords), "\n")

    overlaps <- findOverlaps(peak_gr, gene_coords)
    cat("Number of peak-gene overlaps:", length(overlaps), "\n")

    gene_activity_df <- summarize_one_peak_matrix(
        peak_input_path = input_path,
        peak_gr = peak_gr,
        gene_coords = gene_coords,
        overlaps = overlaps
    )

    input_trans_gene_activity_df <- summarize_one_peak_matrix(
        peak_input_path = INPUT_TRANS_PATH,
        peak_gr = peak_gr,
        gene_coords = gene_coords,
        overlaps = overlaps
    )

    cat("Reading ground truth:", GROUND_TRUTH_path, "\n")
    ground_truth_df <- read_feather(GROUND_TRUTH_path)
    ground_truth_df <- as.data.frame(ground_truth_df)

    if (ncol(ground_truth_df) < 2) {
        stop("GROUND_TRUTH file must contain one cell/barcode column and gene columns.")
    }

    gt_cell_col <- colnames(ground_truth_df)[1]

    common_cells <- Reduce(
        intersect,
        list(
            gene_activity_df[[1]],
            input_trans_gene_activity_df[[1]],
            ground_truth_df[[gt_cell_col]]
        )
    )

    common_genes <- Reduce(
        intersect,
        list(
            colnames(gene_activity_df)[-1],
            colnames(input_trans_gene_activity_df)[-1],
            colnames(ground_truth_df)[-1]
        )
    )

    cat("Common cells across recon, input_trans, and ground truth:", length(common_cells), "\n")
    cat("Common genes across recon, input_trans, and ground truth:", length(common_genes), "\n")

    if (length(common_cells) == 0) {
        stop("No common cells found across reconstruction, input transformed matrix, and ground truth.")
    }

    if (length(common_genes) == 0) {
        stop("No common genes found across reconstruction, input transformed matrix, and ground truth.")
    }

    gene_activity_df <- gene_activity_df[
        match(common_cells, gene_activity_df[[1]]),
        c(colnames(gene_activity_df)[1], common_genes),
        drop = FALSE
    ]

    input_trans_gene_activity_df <- input_trans_gene_activity_df[
        match(common_cells, input_trans_gene_activity_df[[1]]),
        c(colnames(input_trans_gene_activity_df)[1], common_genes),
        drop = FALSE
    ]

    ground_truth_df <- ground_truth_df[
        match(common_cells, ground_truth_df[[gt_cell_col]]),
        c(gt_cell_col, common_genes),
        drop = FALSE
    ]

    colnames(gene_activity_df)[1] <- "pos"
    colnames(input_trans_gene_activity_df)[1] <- "pos"
    colnames(ground_truth_df)[1] <- "pos"

    if (!identical(gene_activity_df$pos, input_trans_gene_activity_df$pos)) {
        stop("Cell order mismatch between summarized reconstruction and summarized input transformed matrix.")
    }

    if (!identical(gene_activity_df$pos, ground_truth_df$pos)) {
        stop("Cell order mismatch between summarized reconstruction and filtered ground truth.")
    }

    if (!identical(colnames(gene_activity_df), colnames(input_trans_gene_activity_df))) {
        stop("Column order mismatch between summarized reconstruction and summarized input transformed matrix.")
    }

    if (!identical(colnames(gene_activity_df), colnames(ground_truth_df))) {
        stop("Column order mismatch between summarized reconstruction and filtered ground truth.")
    }

    out_dir1 <- dirname(output_recon_path)
    if (!dir.exists(out_dir1)) {
        dir.create(out_dir1, recursive = TRUE, showWarnings = FALSE)
    }

    out_dir2 <- dirname(INPUT_TRANS_PATH_filtered)
    if (!dir.exists(out_dir2)) {
        dir.create(out_dir2, recursive = TRUE, showWarnings = FALSE)
    }

    out_dir3 <- dirname(filtered_GROUND_TRUTH_path)
    if (!dir.exists(out_dir3)) {
        dir.create(out_dir3, recursive = TRUE, showWarnings = FALSE)
    }

    write_feather(gene_activity_df, output_recon_path)
    write_feather(input_trans_gene_activity_df, INPUT_TRANS_PATH_filtered)
    write_feather(ground_truth_df, filtered_GROUND_TRUTH_path)

    cat("Saved summarized gene activity reconstruction to:", output_recon_path, "\n")
    cat("Saved summarized and filtered input transformed matrix to:", INPUT_TRANS_PATH_filtered, "\n")
    cat("Saved filtered ground truth to:", filtered_GROUND_TRUTH_path, "\n")
    cat("Final matched dim:", nrow(gene_activity_df), "cells x", ncol(gene_activity_df) - 1, "genes\n")
}