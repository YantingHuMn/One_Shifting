select_best_norm <- function(csv_path) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(dplyr)
    })

    df <- read.csv(csv_path)

    result <- df %>%
        group_by(REP1_trans) %>%
        mutate(
            x_mean_rank = rank(-x_mean, ties.method = "min"),
            residual_rank = rank(residual_mean, ties.method = "min"),
            combined_rank = x_mean_rank + residual_rank,
            tiebreaker = abs(abs(x_mean) - abs(residual_mean))
        ) %>%
        arrange(REP1_trans, combined_rank, tiebreaker)

    print(result %>% select(REP1_trans, REP1_norm, x_mean, x_mean_rank, residual_mean, residual_rank, combined_rank, tiebreaker), n = Inf)

    best <- result %>%
        group_by(REP1_trans) %>%
        slice(1)

    best_out <- best %>%
        ungroup() %>%
        select(REP1_trans, REP1_norm, x_mean, x_mean_rank, residual_mean, residual_rank, combined_rank)

    out_path <- sub("\\.csv$", "_best_norm.csv", csv_path)
    write.csv(best_out, out_path, row.names = FALSE)

    return(out_path)
}

