# Match BAM Files
# Post: Match BAM files in a directory to a vector of filtered sample names, handling "_downsampled" suffix mismatches automatically.
# Parameter:
#   bam_path      : Directory path containing BAM files (searched recursively).
#   filtered_crf  : Character vector of sample names to match against.
# Output: Character vector of full BAM file paths that match the filtered sample names.
match_bam_files <- function(bam_path, filtered_crf) {
    bam_files <- list.files(path = bam_path, pattern = "\\.bam$", recursive = TRUE, full.names = TRUE)
    bam_files <- bam_files[!grepl("unknown|IgG|is", bam_files, ignore.case = TRUE)]
    bam_names <- tools::file_path_sans_ext(basename(bam_files))

    direct_match <- sum(bam_names %in% filtered_crf)

    if (direct_match == 0 && any(grepl("_downsampled$", bam_names))) {
        matched_crf <- paste0(filtered_crf, "_downsampled")
        cat("Bam files have '_downsampled' suffix, appending to filtered_crf\n")
    } else if (direct_match == 0 && any(grepl("_downsampled$", filtered_crf))) {
        matched_crf <- gsub("_downsampled$", "", filtered_crf)
        cat("Bam files don't have '_downsampled', stripping from filtered_crf\n")
    } else {
        matched_crf <- filtered_crf
    }

    filtered_bam_files <- bam_files[bam_names %in% matched_crf]
    cat("Matched bam files:", length(filtered_bam_files), "\n")
    return(filtered_bam_files)
}