create_hex_plot <- function(plot_data, df1_x_mean, df2_y_mean, 
                            residual_sum, residual_mean, n_points, above_line, below_line, 
                            on_line, trans_val, norm_val, metric_type, V1, y_label, x_label) {
    
    subtitle_line1 <- paste0("x_m: ", round(df1_x_mean, 4), 
                             ", y_m: ", round(df2_y_mean, 4),
                             ", Res_s: ", round(residual_sum, 4),
                             ", Res_m: ", round(residual_mean, 4),
                             " (n = ", n_points, " pairs)")
    subtitle_line2 <- paste0("Above: ", above_line, ", Below: ", below_line, ", On line: ", on_line)
    subtitle_text <- paste0(subtitle_line1, "\n", subtitle_line2)
    
    p <- ggplot(plot_data, aes(x = df1_VAE, y = df2_VAE)) +
        geom_hex(bins = 50) +
        scale_fill_viridis_c() +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
        labs(x = paste0(x_label, "_INPUT_", V1),          
             y = y_label,         
             title = paste0(metric_type, ": ", V1, " ", trans_val, "; ", norm_val),
             subtitle = subtitle_text) +
        scale_x_continuous(limits = c(-0.5, 1)) +
        scale_y_continuous(limits = c(-0.5, 1)) +
        theme_bw() +
        theme(
            plot.subtitle = element_text(size = 9, color = "gray30"),
            legend.key.width = unit(0.3, "cm")
        )
    
    return(p)
}