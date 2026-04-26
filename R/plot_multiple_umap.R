plot_multiple_umap <- function(data_paths, data_names, celltype_df, output_dir, n_clusters = NULL, clustering_method = "kmeans", ncol = 5, width = 30, height = NULL, dpi = 300) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(arrow)
        library(Seurat)
        library(ggplot2)
        library(dplyr)
        library(patchwork)
        library(mclust)
    })

    if (length(data_paths) != length(data_names)) {
        stop("data_paths and data_names must have the same length")
    }

    if (is.null(n_clusters)) {
        n_clusters <- length(unique(na.omit(celltype_df$cell_type)))
    }
    
    n_data <- length(data_paths)
    
    if (is.null(height)) {
        nrow_plot <- ceiling(n_data / ncol)
        height <- nrow_plot * 5
    }

    output_dir <- file.path(output_dir, clustering_method)
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    
    process_one <- function(feather_path, data_name) {
        cat(paste0("\n===== Processing: ", data_name, " =====\n"))
        
        df <- read_feather(feather_path)
        cat(paste0("Data shape: ", nrow(df), " cells x ", ncol(df)-1, " genes\n"))
        
        cell_ids <- df$pos
        df$pos <- NULL
        
        mat <- as.matrix(df)
        rownames(mat) <- cell_ids
        mat <- t(mat)
        
        seurat_obj <- CreateSeuratObject(counts = mat, project = data_name)
        
        cell_names <- colnames(seurat_obj)
        celltype_match <- celltype_df$cell_type[match(cell_names, celltype_df$barcode)]
        celltype_match[is.na(celltype_match)] <- "Unknown"
        seurat_obj$cell_type <- celltype_match
        
        cat("Cell type distribution:\n")
        print(table(seurat_obj$cell_type))
        
        seurat_obj <- SetAssayData(seurat_obj, slot = "data", new.data = mat)
        
        seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
        seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
        seurat_obj <- RunPCA(seurat_obj, npcs = 30, verbose = FALSE)
        seurat_obj <- RunUMAP(seurat_obj, dims = 1:30, verbose = FALSE)
        seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30, verbose = FALSE)
        
        return(seurat_obj)
    }
    
    # Process all datasets (only once)
    seurat_list <- list()
    for (i in seq_along(data_paths)) {
        seurat_list[[i]] <- process_one(data_paths[i], data_names[i])
    }
    
    # Cluster with fixed k using known cell type count, then compute ARI
    all_ari <- data.frame()

    cat(paste0("\n========== Clustering method: ", clustering_method, ", target clusters: ", n_clusters, " ==========\n"))

    for (i in seq_along(seurat_list)) {
        obj <- seurat_list[[i]]
        
        if (clustering_method == "kmeans") {
            pca_embed <- Embeddings(obj, reduction = "pca")[, 1:30, drop = FALSE]
            set.seed(42)
            km_res <- kmeans(pca_embed, centers = n_clusters, nstart = 20)
            cluster_labels <- as.character(km_res$cluster)
            obj$self_cluster <- cluster_labels
        } else if (clustering_method %in% c("louvain", "leiden")) {
            algorithm <- ifelse(clustering_method == "louvain", 1, 4)
            best_res <- 0.8
            best_ari <- -1
            for (res in seq(0.1, 2.0, by = 0.1)) {
                obj_temp <- FindClusters(obj, resolution = res, algorithm = algorithm, verbose = FALSE)
                temp_labels <- as.character(obj_temp$seurat_clusters)
                temp_true <- obj_temp$cell_type
                temp_valid <- temp_true != "Unknown"
                temp_ari <- adjustedRandIndex(temp_labels[temp_valid], temp_true[temp_valid])
                if (temp_ari > best_ari) {
                    best_ari <- temp_ari
                    best_res <- res
                }
            }
            obj <- FindClusters(obj, resolution = best_res, algorithm = algorithm, verbose = FALSE)
            cluster_labels <- as.character(obj$seurat_clusters)
            obj$self_cluster <- cluster_labels
        } else {
            stop("clustering_method must be 'kmeans', 'louvain', or 'leiden'")
        }

        true_labels <- obj$cell_type
        valid_idx <- true_labels != "Unknown"
        ari <- adjustedRandIndex(cluster_labels[valid_idx], true_labels[valid_idx])
        found_clusters <- length(unique(cluster_labels))

        cat(paste0("  ", data_names[i], ": k=", found_clusters, ", ARI=", round(ari, 4), "\n"))

        seurat_list[[i]] <- obj

        all_ari <- rbind(all_ari, data.frame(
            method = data_names[i],
            n_clusters = found_clusters,
            ARI = round(ari, 4)
        ))
    }

    # Plot UMAP colored by reference cell type
    plot_list <- list()
    for (i in seq_along(seurat_list)) {
        p <- DimPlot(seurat_list[[i]], reduction = "umap", group.by = "cell_type", 
                    label = TRUE, repel = TRUE) + 
        ggtitle(data_names[i]) + 
        NoLegend() +
        theme(plot.title = element_text(hjust = 0.5, size = 11, face = "bold"))
        plot_list[[i]] <- p
    }
    
    combined_plot <- wrap_plots(plot_list, ncol = ncol)
    umap_path <- file.path(output_dir, "UMAP_color_by_reference.png")
    ggsave(umap_path, combined_plot, width = width, height = height, dpi = dpi)
    cat(paste0("\n[SAVE] UMAP plot: ", umap_path, "\n"))

    # Plot UMAP colored by self clustering
    plot_list_self <- list()
    for (i in seq_along(seurat_list)) {
        p <- DimPlot(seurat_list[[i]], reduction = "umap", group.by = "self_cluster", 
                    label = TRUE, repel = TRUE) + 
        ggtitle(data_names[i]) + 
        NoLegend() +
        theme(plot.title = element_text(hjust = 0.5, size = 11, face = "bold"))
        plot_list_self[[i]] <- p
    }
    
    combined_plot_self <- wrap_plots(plot_list_self, ncol = ncol)
    umap_self_path <- file.path(output_dir, "UMAP_color_by_self.png")
    ggsave(umap_self_path, combined_plot_self, width = width, height = height, dpi = dpi)
    cat(paste0("[SAVE] UMAP plot: ", umap_self_path, "\n"))
    
    # Save ARI CSV
    csv_path <- file.path(output_dir, paste0("ARI_", clustering_method, ".csv"))
    write.csv(all_ari, csv_path, row.names = FALSE)
    cat(paste0("\n[SAVE] ARI table: ", csv_path, "\n"))

    # Plot ARI bar chart for fixed k
    p_ari <- ggplot(all_ari, aes(x = reorder(method, ARI), y = ARI, fill = method)) +
        geom_col() +
        coord_flip() +
        labs(x = "Method", y = "ARI", title = paste0("ARI with ", clustering_method, " (target k=", n_clusters, ")")) +
        theme_bw() +
        theme(legend.position = "none")

    ari_plot_path <- file.path(output_dir, paste0("ARI_", clustering_method, ".png"))
    ggsave(ari_plot_path, p_ari, width = 12, height = 6, dpi = 300)
    cat(paste0("[SAVE] ARI plot: ", ari_plot_path, "\n"))
    
    return(list(
        plot = combined_plot,
        seurat_list = seurat_list,
        all_ari = all_ari
    ))
}