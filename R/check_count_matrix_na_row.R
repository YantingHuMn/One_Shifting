#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop(
    "Usage: Rscript check_feather_complete.R ",
    "<sample_name> <read_dir> <output_csv>\n"
  )
}

sample_name <- args[1]
read_dir <- args[2]
output_csv <- args[3]


count_na_rows <- function(df) {

  row_has_na <- rep(FALSE, nrow(df))

  for (column_name in colnames(df)) {

    x <- df[[column_name]]

    if (is.numeric(x)) {
      x <- x + 0
    }

    row_has_na <- row_has_na | is.na(x)
  }

  sum(row_has_na)
}


count_recon_feather <- function(method_dir) {

  if (!dir.exists(method_dir)) {
    return(0L)
  }

  length(
    list.files(
      method_dir,
      pattern = "^reconstruct.*\\.feather$",
      full.names = TRUE,
      recursive = TRUE
    )
  )
}


check_modality <- function(sample_name, modality, read_dir) {

  input_dir <- file.path(
    read_dir,
    paste0(sample_name, "_INPUT_", modality)
  )

  feather_dir <- file.path(
    input_dir,
    paste0(sample_name, "_", modality)
  )

  # Report only; these counts do not affect problem or note.
  method_counts <- data.frame(
    DCA_mse = count_recon_feather(
      file.path(input_dir, "DCA_mse")
    ),
    VAE = count_recon_feather(
      file.path(input_dir, "VAE")
    ),
    scVI_mse = count_recon_feather(
      file.path(input_dir, "scVI_mse")
    ),
    Transformer_denoise = count_recon_feather(
      file.path(input_dir, "Transformer_denoise")
    )
  )

  note_parts <- character(0)
  has_problem <- FALSE

  if (!dir.exists(feather_dir)) {

    return(cbind(
      data.frame(
        sample_name = sample_name,
        modality = modality,
        directory = feather_dir,
        n_feather = NA_integer_,
        problem = "yes",
        note = "directory_not_found",
        stringsAsFactors = FALSE
      ),
      method_counts
    ))
  }

  feather_files <- sort(
    list.files(
      feather_dir,
      pattern = "\\.feather$",
      full.names = TRUE,
      recursive = FALSE
    )
  )

  n_feather <- length(feather_files)

  # Check whether exactly six feather files are present.
  if (n_feather != 6) {

    has_problem <- TRUE

    note_parts <- c(
      note_parts,
      paste0(
        "feather_count=",
        n_feather,
        " (expected 6)"
      )
    )
  }

  # Check each feather file for rows containing at least one NA/NaN.
  if (n_feather > 0) {

    for (feather_file in feather_files) {

      read_result <- tryCatch(
        arrow::read_feather(feather_file),
        error = function(e) e
      )

      if (inherits(read_result, "error")) {

        has_problem <- TRUE

        note_parts <- c(
          note_parts,
          paste0(
            basename(feather_file),
            ": cannot_read"
          )
        )

        next
      }

      n_na_rows <- count_na_rows(read_result)

      if (n_na_rows > 0) {

        has_problem <- TRUE

        note_parts <- c(
          note_parts,
          paste0(
            basename(feather_file),
            ": ",
            n_na_rows,
            " NA rows"
          )
        )
      }
    }
  }

  cbind(
    data.frame(
      sample_name = sample_name,
      modality = modality,
      directory = feather_dir,
      n_feather = n_feather,
      problem = if (has_problem) "yes" else "no",
      note = if (length(note_parts) > 0) {
        paste(note_parts, collapse = "; ")
      } else {
        NA_character_
      },
      stringsAsFactors = FALSE
    ),
    method_counts
  )
}


# Every sample always produces exactly two rows.
result <- rbind(
  check_modality(
    sample_name = sample_name,
    modality = "RNA",
    read_dir = read_dir
  ),
  check_modality(
    sample_name = sample_name,
    modality = "ATAC",
    read_dir = read_dir
  )
)


output_dir <- dirname(output_csv)

if (!dir.exists(output_dir)) {
  dir.create(
    output_dir,
    recursive = TRUE
  )
}


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


print(result)