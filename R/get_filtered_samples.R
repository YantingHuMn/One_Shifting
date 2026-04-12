get_filtered_samples <- function(path1, filtered_percentile = 0.25) {
    bam_files <- list.files(path = path1, pattern = "\\.bam$", recursive = TRUE, full.names = TRUE)
    bam_files <- bam_files[!grepl("unknown|IgG|is", bam_files, ignore.case = TRUE)]
    result <- qc(file_paths = bam_files, filtered_percentile = filtered_percentile, output_path_dir = NULL, save = FALSE)
    return(result$filtered_crf)
}
