# Function to create a "No Data" plot
create_no_data_plot <- function() {
    ggplot() +
    ggtitle("No Data") +
    theme_void() +
    theme(plot.title = element_text(hjust = 0.5, size = 8, margin = margin(b = 2)))
}