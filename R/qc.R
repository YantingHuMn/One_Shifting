# Qualification Control
# Post: Perform QC by counting reads/peaks from input BAM/BED files.
# Parameter:
#   file_paths         : Character vector of BAM or BED file paths.
#   filtered_percentile: Numeric in (0, 1); percentile threshold for filtering low-count samples. Default 0.25.
#   output_path_dir    : Directory to save output CSVs if save == TRUE.
#   save               : Logical. If TRUE, save full and filtered count tables as CSV in output_path_dir. Default TRUE.
# Output: A list containing:
#   - all_df       : Data frame of all files and their read/peak counts.
#   - filtered_df  : Data frame after filtering by read count threshold.
#   - filtered_crf : Character vector of file names passing the filter.
#   - total_reads  : Total read/peak count across all files.
qc <- function(file_paths, filtered_percentile = 0.25, output_path_dir = NULL, save = TRUE) {
    # load library
    suppressPackageStartupMessages({
        library(ChIPseeker)
        library(ComplexHeatmap)
        library(glue)
        library(latex2exp)
        library(Rsamtools)
        library(GenomicAlignments)
    })

    ext <- tools::file_ext(file_paths)

    df <- data.frame(file = character(), read_count = numeric(), stringsAsFactors = FALSE)

    for (k in seq_along(file_paths)) {
        file_path <- file_paths[k]
        file_name <- basename(tools::file_path_sans_ext(file_path))
        if (all(ext == "bam")) {
            temp <- readGAlignmentPairs(file_path)
            count <- length(temp)
        } else if (all(ext == "bed") && file.exists(file_path) && length(file_path) > 0 && file.size(file_path) > 0) {
            peak <- ChIPseeker::readPeakFile(file_path, as = "GRanges")   # Use ChIPseeker to read peak files
            peak_count <- length(peak)
            count <- peak_count
        }
        df <- rbind(df, data.frame(file = file_name, read_count = count, stringsAsFactors = FALSE)) # return: total df
    }

    threshold <- quantile(as.numeric(df$read_count), probs = filtered_percentile, type = 3)
    filtered_df <- df[df$read_count >= threshold, ]
    all_df <- df
    all_df$file <- sub("\\.[^\\.]+$", "", all_df$file) # To-do: double check output file name of peak calling reuslts, currently we only support dot(.) as a seperator
    filtered_df$file <- sub("\\.[^\\.]+$", "", filtered_df$file)
    filtered_df_vector <- filtered_df$file # return: filtered vector
    total_reads <- sum(df$read_count) # return: total reads

    # save
    if (save == TRUE) {
        if (is.null(output_path_dir)) {
            stop("Error: Need output dir path: 'output_path_dir' is NULL.")
        }
        
        if (!dir.exists(output_path_dir)) {
            dir.create(output_path_dir, recursive = TRUE)
        }

        write.csv(all_df, file = file.path(output_path_dir, paste0("all_read_counts_", ext[1], ".csv")), row.names = FALSE)
        write.csv(filtered_df, file = file.path(output_path_dir, paste0("filtered_read_counts_", ext[1], ".csv")), row.names = FALSE)
        writeLines(as.character(total_reads), con = file.path(output_path_dir, paste0("total_reads_", ext[1], ".txt")))
    }

    return (list(all_df = all_df, filtered_df = filtered_df, filtered_crf = filtered_df_vector, total_reads = total_reads))
}