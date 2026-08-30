#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop(
    "Usage: Rscript check_sample_dimension.R <sample_name> <read_dir> <output_csv>\n"
  )
}

sample_name <- args[1]
read_dir <- args[2]
output_csv <- args[3]

rna_file <- file.path(
  read_dir,
  paste0(sample_name, "_INPUT_RNA"),
  paste0(sample_name, "_RNA"),
  "Count_Matrix_norm_by_no_norm.feather"
)

atac_file <- file.path(
  read_dir,
  paste0(sample_name, "_INPUT_ATAC"),
  paste0(sample_name, "_ATAC"),
  "Count_Matrix_norm_by_no_norm.feather"
)

check_one_file <- function(file_path, sample_name, modality) {

  if (!file.exists(file_path)) {
    warning("File not found: ", file_path)

    return(data.frame(
      sample_name = sample_name,
      modality = modality,
      nrow = NA_integer_,
      ncol = NA_integer_,
      first_col_is_pos = NA,
      stringsAsFactors = FALSE
    ))
  }

  df <- arrow::read_feather(file_path)

  first_col_is_pos <- ncol(df) > 0 && colnames(df)[1] == "pos"

  data.frame(
    sample_name = sample_name,
    modality = modality,
    nrow = nrow(df),
    ncol = ncol(df),
    first_col_is_pos = first_col_is_pos,
    stringsAsFactors = FALSE
  )
}

rna_result <- check_one_file(
  rna_file,
  sample_name,
  "RNA"
)

atac_result <- check_one_file(
  atac_file,
  sample_name,
  "ATAC"
)

result <- rbind(
  rna_result,
  atac_result
)

output_dir <- dirname(output_csv)

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

if (!file.exists(output_csv)) {

  write.csv(
    result,
    output_csv,
    row.names = FALSE
  )

  cat("Created:", output_csv, "\n")

} else {

  write.table(
    result,
    output_csv,
    sep = ",",
    row.names = FALSE,
    col.names = FALSE,
    append = TRUE
  )

  cat("Appended to:", output_csv, "\n")
}

print(result)