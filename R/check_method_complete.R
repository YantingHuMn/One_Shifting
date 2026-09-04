#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop(
    "Usage: Rscript check_method_complete.R <sample_name> <read_dir> <output_csv>\n"
  )
}

sample_name <- args[1]
read_dir <- args[2]
output_csv <- args[3]

check_modality <- function(sample_name, modality, read_dir) {

  input_dir <- file.path(
    read_dir,
    paste0(sample_name, "_INPUT_", modality)
  )

  if (!dir.exists(input_dir)) {
    return(data.frame(
      sample_name = sample_name,
      modality = modality,
      problem = "input_dir_not_found",
      file = NA_character_,
      nrow = NA_integer_,
      n_unique_method = NA_integer_,
      note = NA_character_,
      stringsAsFactors = FALSE
    ))
  }

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

  csv_files <- file.path(
    input_dir,
    expected_csv_names
  )

  missing_files <- csv_files[!file.exists(csv_files)]

  if (length(missing_files) > 0) {
    return(data.frame(
      sample_name = sample_name,
      modality = modality,
      problem = paste0("missing_", length(missing_files), "_expected_csv"),
      file = paste(basename(missing_files), collapse = ";"),
      nrow = NA_integer_,
      n_unique_method = NA_integer_,
      note = NA_character_,
      stringsAsFactors = FALSE
    ))
  }

  problems <- list()

  for (csv_file in csv_files) {

    df <- tryCatch(
      read.csv(csv_file, stringsAsFactors = FALSE),
      error = function(e) NULL
    )

    if (is.null(df)) {
      problems[[length(problems) + 1]] <- data.frame(
        sample_name = sample_name,
        modality = modality,
        problem = "cannot_read_csv",
        file = basename(csv_file),
        nrow = NA_integer_,
        n_unique_method = NA_integer_,
        note = NA_character_,
        stringsAsFactors = FALSE
      )

      next
    }

    n_row <- nrow(df)

    if (!"method" %in% colnames(df)) {
      problems[[length(problems) + 1]] <- data.frame(
        sample_name = sample_name,
        modality = modality,
        problem = "method_column_missing",
        file = basename(csv_file),
        nrow = n_row,
        n_unique_method = NA_integer_,
        note = NA_character_,
        stringsAsFactors = FALSE
      )

      next
    }

    n_method <- length(unique(df$method))

    note_text <- character(0)

    expected_metrics <- c("pearson", "spearman")
    expected_trans <- c(
      "no_trans",
      "sqrt",
      "sqrt+1",
      "log2",
      "count+1",
      "log2(count+2)"
    )

    if (all(c("metric", "trans") %in% colnames(df))) {

      for (method_name in unique(df$method)) {

        for (metric_name in expected_metrics) {

          tmp <- df[
            df$method == method_name &
              df$metric == metric_name,
            ,
            drop = FALSE
          ]

          if (nrow(tmp) == 0) {

            note_text <- c(
              note_text,
              paste0(
                method_name,
                ": missing ",
                metric_name
              )
            )

          } else {

            missing_trans <- setdiff(
              expected_trans,
              unique(tmp$trans)
            )

            if (length(missing_trans) > 0) {

              note_text <- c(
                note_text,
                paste0(
                  method_name,
                  ": ",
                  metric_name,
                  " missing ",
                  paste(missing_trans, collapse = ",")
                )
              )
            }
          }
        }
      }
    }

    if (n_row != 46 || n_method != 4 || length(note_text) > 0) {

      problem_text <- paste(
        c(
          if (n_row != 46) paste0("nrow=", n_row) else NULL,
          if (n_method != 4) paste0("unique_method=", n_method) else NULL
        ),
        collapse = ";"
      )

      problems[[length(problems) + 1]] <- data.frame(
        sample_name = sample_name,
        modality = modality,
        problem = problem_text,
        file = basename(csv_file),
        nrow = n_row,
        n_unique_method = n_method,
        note = if (length(note_text) > 0) {
          paste(note_text, collapse = "; ")
        } else {
          NA_character_
        },
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(problems) == 0) {
    return(NULL)
  }

  do.call(rbind, problems)
}

rna_problem <- check_modality(
  sample_name,
  "RNA",
  read_dir
)

atac_problem <- check_modality(
  sample_name,
  "ATAC",
  read_dir
)

result <- rbind(
  rna_problem,
  atac_problem
)

output_dir <- dirname(output_csv)

if (!dir.exists(output_dir)) {
  dir.create(
    output_dir,
    recursive = TRUE
  )
}

if (!is.null(result) && nrow(result) > 0) {

  if (!file.exists(output_csv)) {

    write.csv(
      result,
      output_csv,
      row.names = FALSE
    )

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

  cat(
    "Problems found for sample:",
    sample_name,
    "\n"
  )

  print(result)

} else {

  cat(
    "All expected CSV files are complete for RNA and ATAC:",
    sample_name,
    "\n"
  )
}