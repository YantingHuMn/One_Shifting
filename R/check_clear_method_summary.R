#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop(
    "Usage: Rscript remove_duplicate_summary.R <sample_name> <read_dir> <output_csv>\n"
  )
}

sample_name <- args[1]
read_dir <- args[2]
output_csv <- args[3]

process_modality <- function(sample_name, modality, read_dir) {

  input_dir <- file.path(
    read_dir,
    paste0(sample_name, "_INPUT_", modality)
  )

  expected_csv_names <- c(
    "bubble_plot_summary_col_gene_v1_reverse_v2_no_trans.csv",
    "bubble_plot_summary_col_gene_v1_trans_v2_no_trans.csv",
    "bubble_plot_summary_col_gene_v1_trans_v2_trans_norm_100000.csv",
    "bubble_plot_summary_col_gene_v1_trans_v2_trans.csv",
    "bubble_plot_summary_row_cell_v1_reverse_v2_no_trans.csv",
    "bubble_plot_summary_row_cell_v1_trans_v2_no_trans.csv",
    "bubble_plot_summary_row_cell_v1_trans_v2_trans_norm_100000.csv",
    "bubble_plot_summary_row_cell_v1_trans_v2_trans.csv"
  )

  results <- list()

  for (csv_name in expected_csv_names) {

    csv_file <- file.path(input_dir, csv_name)
    status <- "unchanged"
    n_before <- NA_integer_
    n_after <- NA_integer_
    n_removed <- NA_integer_
    note <- NA_character_

    if (!file.exists(csv_file)) {

      status <- "file_not_found"

    } else {

      tryCatch({

        df <- read.csv(csv_file, stringsAsFactors = FALSE)

        key_cols <- c("method", "data_type", "metric", "trans")

        if (ncol(df) < 4 ||
            !identical(colnames(df)[1:4], key_cols)) {
          stop("First four columns must be: method, data_type, metric, trans")
        }

        n_before <- nrow(df)

        # Keep the last occurrence of each key, preserving remaining row order.
        keep <- !duplicated(df[, 1:4, drop = FALSE], fromLast = TRUE)
        clean_df <- df[keep, , drop = FALSE]

        n_after <- nrow(clean_df)
        n_removed <- sum(!keep)

        if (n_removed > 0) {

          # Use a unique backup filename; never overwrite an existing backup.
          backup_base <- paste0(
            csv_file,
            ".bak_",
            format(Sys.time(), "%Y%m%d_%H%M%S"),
            "_",
            Sys.getpid()
          )

          backup_file <- backup_base
          suffix <- 0L

          while (file.exists(backup_file)) {
            suffix <- suffix + 1L
            backup_file <- paste0(backup_base, "_", suffix)
          }

          if (!file.copy(csv_file, backup_file, overwrite = FALSE)) {
            stop("Backup failed")
          }

          # Write to a temporary file before replacing the original.
          temp_file <- tempfile(
            pattern = "deduplicated_",
            tmpdir = input_dir,
            fileext = ".csv"
          )

          tryCatch({

            write.csv(clean_df, temp_file, row.names = FALSE)

            if (!file.rename(temp_file, csv_file)) {
              stop("Cannot replace original CSV")
            }

          }, finally = {
            if (file.exists(temp_file)) {
              unlink(temp_file)
            }
          })

          status <- "deduplicated"
          note <- paste0("backup=", basename(backup_file))
        }

      }, error = function(e) {

        status <<- "error"
        note <<- conditionMessage(e)

      })
    }

    results[[length(results) + 1]] <- data.frame(
      sample_name = sample_name,
      modality = modality,
      file = csv_name,
      status = status,
      n_before = n_before,
      n_after = n_after,
      n_removed = n_removed,
      note = note,
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, results)
}

result <- rbind(
  process_modality(sample_name, "RNA", read_dir),
  process_modality(sample_name, "ATAC", read_dir)
)

output_dir <- dirname(output_csv)

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

if (!file.exists(output_csv)) {

  write.csv(result, output_csv, row.names = FALSE)

} else {

  write.table(
    result,
    output_csv,
    sep = ",",
    row.names = FALSE,
    col.names = FALSE,
    append = TRUE
  )
}

print(result)