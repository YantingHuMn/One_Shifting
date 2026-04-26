
append_registry <- function(registry_file, path, method, trans, norm_factor) {
    row <- data.frame(
        path = path,
        method = method,
        trans = trans,
        norm_factor = norm_factor,
        stringsAsFactors = FALSE
    )
    
    if (file.exists(registry_file)) {
        write.table(row, registry_file, append = TRUE, sep = ",",
                    col.names = FALSE, row.names = FALSE, quote = TRUE)
    } else {
        write.csv(row, registry_file, row.names = FALSE)
    }
}

args <- commandArgs(trailingOnly = TRUE)
registry_file <- args[1]
path <- args[2]
method <- args[3]
trans <- args[4]
norm_factor <- args[5]

append_registry(registry_file = registry_file, path = path, method = method, trans = trans, norm_factor = norm_factor)
