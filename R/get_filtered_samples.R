# Get Filtered Samples
# Post: List BAM files in a directory, exclude unwanted files (unknown, IgG, input), run QC, and return names of samples passing the read-count threshold.
# Parameter:
#   path1               : Directory path containing BAM files (searched recursively).
#   filtered_percentile : Numeric in (0, 1); percentile threshold for QC filtering. Default 0.25.
# Output: Character vector of sample names (without extension) that pass the QC filter.
get_filtered_samples <- function(path1, filtered_percentile = 0.25) {
    bam_files <- list.files(path = path1, pattern = "\\.bam$", recursive = TRUE, full.names = TRUE)
    bam_files <- bam_files[!grepl("unknown|IgG|is", bam_files, ignore.case = TRUE)]
    result <- qc(file_paths = bam_files, filtered_percentile = filtered_percentile, output_path_dir = NULL, save = FALSE)
    return(result$filtered_crf)
}
