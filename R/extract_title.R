# Function: Extract transformation information
extract_title <- function(filename) {
    basename_file <- basename(filename)
    name_without_ext <- sub("\\.[^.]*$", "", basename_file)
    clean_name <- sub("^[dgj]_(pearson|spearman|roc)_(scatter|curve)", "", name_without_ext)

    v1_to_v2_match <- regmatches(clean_name, regexpr("v1_(.*?)_v2", clean_name))

    if (length(v1_to_v2_match) > 0) {
    transformation <- sub("v1|(v2)", "", v1_to_v2_match)
    split_transformation <- unlist(strsplit(transformation, "_"))
    split_transformation <- split_transformation[split_transformation != "v2"]
    split_transformation <- split_transformation[split_transformation != ""]

    if (length(split_transformation) > 4) {
        position <- which(split_transformation == "no")
        if (length(position) > 0 && position[1] == 4) {
        v1_norm <- paste(split_transformation[4], split_transformation[5], sep = "_")
        v1_trans <- split_transformation[2]
        } else if (length(position) > 0 && position[1] == 2) {
        v1_norm <- split_transformation[5]
        v1_trans <- paste(split_transformation[2], split_transformation[3], sep = "_")
        } else {
        v1_norm <- split_transformation[5]
        v1_trans <- split_transformation[2]
        }
    } else if (length(split_transformation) == 4) {
        v1_norm <- split_transformation[4]
        v1_trans <- split_transformation[2]
    } else {
        v1_norm <- "unknown"
        v1_trans <- if (length(split_transformation) >= 2) split_transformation[2] else "unknown"
    }
    } else {
    v1_norm <- "unknown"
    v1_trans <- "unknown"
    }

    title <- paste0("V1: ", v1_trans, ";; norm ", v1_norm)
    return(title)
}