#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop(
    "Usage: Rscript check_baseline_reconstruction.R ",
    "<sample_name> <read_dir> <output_csv>\n"
  )
}

sample_name <- args[1]
read_dir <- args[2]
output_csv <- args[3]

methods <- c("DCA", "scVI", "SAVER")
modalities <- c("RNA", "ATAC")


check_method <- function(sample_name, modality, method, read_dir) {

  method_dir <- file.path(
    read_dir,
    paste0(sample_name, "_INPUT_", modality),
    method
  )

  if (!dir.exists(method_dir)) {

    return(data.frame(
      sample_name = sample_name,
      modality = modality,
      method = method,
      directory = method_dir,
      n_reconstructed_feather = 0L,
      problem = "yes",
      note = "method_directory_not_found",
      stringsAsFactors = FALSE
    ))
  }

  feather_files <- list.files(
    method_dir,
    pattern = "^reconstruct.*\\.feather$",
    full.names = TRUE,
    recursive = TRUE,
    ignore.case = TRUE
  )

  n_feather <- length(feather_files)

  data.frame(
    sample_name = sample_name,
    modality = modality,
    method = method,
    directory = method_dir,
    n_reconstructed_feather = n_feather,
    problem = if (n_feather != 1) "yes" else "no",
    note = if (n_feather == 0) {
      "reconstructed_feather_not_found"
    } else if (n_feather > 1) {
      "multiple_reconstructed_feather_files"
    } else {
      NA_character_
    },
    stringsAsFactors = FALSE
  )
}


result <- do.call(
  rbind,
  lapply(
    modalities,
    function(modality) {
      do.call(
        rbind,
        lapply(
          methods,
          function(method) {
            check_method(sample_name, modality, method, read_dir)
          }
        )
      )
    }
  )
)


output_dir <- dirname(output_csv)

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


if (!file.exists(output_csv)) {

  write.csv(
    result,
    output_csv,
    row.names = FALSE,
    na = ""
  )

} else {

  write.table(
    result,
    output_csv,
    sep = ",",
    row.names = FALSE,
    col.names = FALSE,
    append = TRUE,
    na = ""
  )
}

print(result)