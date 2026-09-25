#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

base_dir <- args[1]
output_csv <- args[2]

suppressPackageStartupMessages({
    library(arrow)
})

read_matrix_info <- function(file_path) {

    tbl <- read_feather(
        file_path,
        as_data_frame = FALSE
    )

    nr <- nrow(tbl)
    nc <- ncol(tbl)
    column_ids <- names(tbl)

    rm(tbl)
    gc(verbose = FALSE)

    # Only materialize the first column containing gene IDs.
    id_df <- read_feather(
        file_path,
        col_select = 1
    )

    gene_ids <- as.character(id_df[[1]])
    cell_ids <- column_ids[-1]

    rm(id_df)
    gc(verbose = FALSE)

    list(
        nrow = nr,
        ncol = nc,
        gene_ids = gene_ids,
        cell_ids = cell_ids
    )
}

directory_names <- basename(
    list.dirs(
        base_dir,
        recursive = FALSE,
        full.names = TRUE
    )
)

raw_directory_names <- directory_names[
    !grepl("_INPUT_", directory_names) &
    grepl("_(ATAC|RNA)$", directory_names)
]

sample_names <- sort(
    unique(
        sub(
            "_(ATAC|RNA)$",
            "",
            raw_directory_names
        )
    )
)

results <- vector(
    "list",
    length(sample_names)
)

for (i in seq_along(sample_names)) {

    sample_name <- sample_names[i]

    atac_file <- file.path(
        base_dir,
        paste0(sample_name, "_ATAC"),
        "activity_counts.feather"
    )

    rna_file <- file.path(
        base_dir,
        paste0(sample_name, "_RNA"),
        "rna_counts.feather"
    )

    cat(
        sprintf(
            "[%d/%d] Checking %s\n",
            i,
            length(sample_names),
            sample_name
        )
    )

    if (!file.exists(atac_file) || !file.exists(rna_file)) {

        missing_files <- c(
            if (!file.exists(atac_file)) {
                "activity_counts.feather"
            },
            if (!file.exists(rna_file)) {
                "rna_counts.feather"
            }
        )

        results[[i]] <- data.frame(
            sample_name = sample_name,
            atac_dim = NA_character_,
            rna_dim = NA_character_,
            dimension_match = NA,
            gene_set_match = NA,
            gene_order_match = NA,
            cell_set_match = NA,
            cell_order_match = NA,
            atac_duplicate_genes = NA_integer_,
            rna_duplicate_genes = NA_integer_,
            atac_duplicate_cells = NA_integer_,
            rna_duplicate_cells = NA_integer_,
            status = "missing_file",
            note = paste(missing_files, collapse = "; "),
            stringsAsFactors = FALSE
        )

        next
    }

    atac_info <- tryCatch(
        read_matrix_info(atac_file),
        error = function(e) e
    )

    rna_info <- tryCatch(
        read_matrix_info(rna_file),
        error = function(e) e
    )

    if (
        inherits(atac_info, "error") ||
        inherits(rna_info, "error")
    ) {

        error_messages <- c(
            if (inherits(atac_info, "error")) {
                paste0("ATAC: ", conditionMessage(atac_info))
            },
            if (inherits(rna_info, "error")) {
                paste0("RNA: ", conditionMessage(rna_info))
            }
        )

        results[[i]] <- data.frame(
            sample_name = sample_name,
            atac_dim = NA_character_,
            rna_dim = NA_character_,
            dimension_match = NA,
            gene_set_match = NA,
            gene_order_match = NA,
            cell_set_match = NA,
            cell_order_match = NA,
            atac_duplicate_genes = NA_integer_,
            rna_duplicate_genes = NA_integer_,
            atac_duplicate_cells = NA_integer_,
            rna_duplicate_cells = NA_integer_,
            status = "read_error",
            note = paste(error_messages, collapse = "; "),
            stringsAsFactors = FALSE
        )

        next
    }

    dimension_match <- (
        atac_info$nrow == rna_info$nrow &&
        atac_info$ncol == rna_info$ncol
    )

    gene_set_match <- setequal(
        atac_info$gene_ids,
        rna_info$gene_ids
    )

    gene_order_match <- identical(
        atac_info$gene_ids,
        rna_info$gene_ids
    )

    cell_set_match <- setequal(
        atac_info$cell_ids,
        rna_info$cell_ids
    )

    cell_order_match <- identical(
        atac_info$cell_ids,
        rna_info$cell_ids
    )

    atac_duplicate_genes <- sum(
        duplicated(atac_info$gene_ids)
    )

    rna_duplicate_genes <- sum(
        duplicated(rna_info$gene_ids)
    )

    atac_duplicate_cells <- sum(
        duplicated(atac_info$cell_ids)
    )

    rna_duplicate_cells <- sum(
        duplicated(rna_info$cell_ids)
    )

    all_match <- (
        dimension_match &&
        gene_order_match &&
        cell_order_match &&
        atac_duplicate_genes == 0 &&
        rna_duplicate_genes == 0 &&
        atac_duplicate_cells == 0 &&
        rna_duplicate_cells == 0
    )

    note_parts <- character(0)

    if (!gene_set_match) {
        note_parts <- c(
            note_parts,
            paste0(
                "genes_only_in_ATAC=",
                length(
                    setdiff(
                        atac_info$gene_ids,
                        rna_info$gene_ids
                    )
                )
            ),
            paste0(
                "genes_only_in_RNA=",
                length(
                    setdiff(
                        rna_info$gene_ids,
                        atac_info$gene_ids
                    )
                )
            )
        )
    }

    if (!cell_set_match) {
        note_parts <- c(
            note_parts,
            paste0(
                "cells_only_in_ATAC=",
                length(
                    setdiff(
                        atac_info$cell_ids,
                        rna_info$cell_ids
                    )
                )
            ),
            paste0(
                "cells_only_in_RNA=",
                length(
                    setdiff(
                        rna_info$cell_ids,
                        atac_info$cell_ids
                    )
                )
            )
        )
    }

    if (
        gene_set_match &&
        !gene_order_match
    ) {
        note_parts <- c(
            note_parts,
            "same_gene_set_but_different_order"
        )
    }

    if (
        cell_set_match &&
        !cell_order_match
    ) {
        note_parts <- c(
            note_parts,
            "same_cell_set_but_different_order"
        )
    }

    results[[i]] <- data.frame(
        sample_name = sample_name,
        atac_dim = paste0(
            atac_info$nrow,
            "x",
            atac_info$ncol
        ),
        rna_dim = paste0(
            rna_info$nrow,
            "x",
            rna_info$ncol
        ),
        dimension_match = dimension_match,
        gene_set_match = gene_set_match,
        gene_order_match = gene_order_match,
        cell_set_match = cell_set_match,
        cell_order_match = cell_order_match,
        atac_duplicate_genes = atac_duplicate_genes,
        rna_duplicate_genes = rna_duplicate_genes,
        atac_duplicate_cells = atac_duplicate_cells,
        rna_duplicate_cells = rna_duplicate_cells,
        status = if (all_match) {
            "matched"
        } else {
            "problem"
        },
        note = if (length(note_parts) > 0) {
            paste(note_parts, collapse = "; ")
        } else {
            NA_character_
        },
        stringsAsFactors = FALSE
    )

    rm(atac_info, rna_info)
    gc(verbose = FALSE)
}

result <- do.call(
    rbind,
    results
)

write.csv(
    result,
    output_csv,
    row.names = FALSE
)

print(result)

cat(
    "\nMatched:",
    sum(result$status == "matched"),
    "\nProblems:",
    sum(result$status == "problem"),
    "\nMissing/read errors:",
    sum(result$status %in% c("missing_file", "read_error")),
    "\n"
)