plot_multiple_umap_arrange <- function(data_paths, data_names, celltype_df, output_dir, n_clusters = NULL, clustering_method = "kmeans", ncol = 5, width = 30, height = NULL, dpi = 300, hvg_gene_path = NULL, use_hvg = TRUE, n_hvg = 2000) {
    # Load Libraries
    suppressPackageStartupMessages({
        library(arrow)
        library(Seurat)
        library(ggplot2)
        library(dplyr)
        library(patchwork)
        library(mclust)
    })

    set.seed(42)

    if (length(data_paths) != length(data_names)) {
        stop("data_paths and data_names must have the same length")
    }

    if (is.null(n_clusters)) {
        n_clusters <- length(unique(na.omit(celltype_df$cell_type)))
    }
    
    n_data <- length(data_paths)
    
    if (is.null(height)) {
        nrow_plot <- ceiling(n_data / ncol)
        height <- min(nrow_plot * 5, 48)  # cap at 48 inches
    }

    output_dir <- file.path(output_dir, clustering_method)
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    
    hvg_genes <- NULL
    if (!is.null(hvg_gene_path) && !is.na(hvg_gene_path) && hvg_gene_path != "") {
        if (!file.exists(hvg_gene_path)) {
            stop("[HVG] hvg_gene_path was provided but does not exist: ", hvg_gene_path)
        }
        hvg_genes <- readLines(hvg_gene_path)
        hvg_genes <- unique(hvg_genes)
        hvg_genes <- hvg_genes[!is.na(hvg_genes) & hvg_genes != ""]
        cat(paste0("[HVG] Loaded ", length(hvg_genes), " genes from: ", hvg_gene_path, "\n"))
    }
    
    run_find_clusters_safe <- function(obj, resolution, algorithm, verbose = FALSE) {
        set.seed(42)
        tryCatch(
            FindClusters(obj, resolution = resolution, algorithm = algorithm, random.seed = 42, verbose = verbose),
            error = function(e) {
                msg <- conditionMessage(e)
                if (algorithm == 4 && grepl("numpy|leidenalg|igraph|reticulate|python|ModuleNotFoundError|No module named", msg, ignore.case = TRUE)) {
                    stop(
                        "Leiden clustering failed. This may be due to the Python environment used by Seurat/reticulate.\n\n",
                        "Original error:\n",
                        msg, "\n\n",
                        "If needed, please install the required Python packages in the environment used by R/reticulate:\n",
                        "  conda install -c conda-forge numpy leidenalg python-igraph\n",
                        "or:\n",
                        "  pip install numpy leidenalg igraph\n\n",
                        "Alternatively, set RETICULATE_PYTHON before running R, for example:\n",
                        "  export RETICULATE_PYTHON=/path/to/python\n",
                        call. = FALSE
                    )
                } else {
                    stop(e)
                }
            }
        )
    }

    process_one <- function(feather_path, data_name) {
        cat(paste0("\n===== Processing: ", data_name, " =====\n"))
        
        df <- read_feather(feather_path)
        cat(paste0("Data shape: ", nrow(df), " cells x ", ncol(df)-1, " genes\n"))
        
        if (!is.null(hvg_genes)) {
            keep_genes <- intersect(hvg_genes, colnames(df))
            if (length(keep_genes) == 0) {
                stop("[HVG] No HVG genes found in dataframe columns: ", feather_path)
            }
            df <- df[, c("pos", keep_genes), drop = FALSE]
            cat(paste0("[HVG] After filtering: ", nrow(df), " cells x ", ncol(df)-1, " genes\n"))
        }
        
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
        
        if (use_hvg) {

            if (!is.null(hvg_genes)) {

                features_for_pca <- intersect(hvg_genes, rownames(seurat_obj))

                if (length(features_for_pca) == 0) {
                    stop("[HVG] No provided HVG genes found in dataframe: ", feather_path)
                }

                cat(paste0(
                    "[HVG] Using provided HVG list: ",
                    length(features_for_pca),
                    " genes\n"
                ))

            } else {

                seurat_obj <- FindVariableFeatures(
                    seurat_obj,
                    selection.method = "vst",
                    nfeatures = n_hvg,
                    verbose = FALSE
                )

                features_for_pca <- VariableFeatures(seurat_obj)

                cat(paste0(
                    "[HVG] Using Seurat-selected HVGs: ",
                    length(features_for_pca),
                    " genes\n"
                ))
            }

        } else {

            features_for_pca <- rownames(seurat_obj)

            cat(paste0(
                "[HVG] Using all genes: ",
                length(features_for_pca),
                " genes\n"
            ))
        }
        
        seurat_obj <- ScaleData(seurat_obj, features = features_for_pca, verbose = FALSE)
        seurat_obj <- RunPCA(seurat_obj, features = features_for_pca, npcs = 30, seed.use = 42, verbose = FALSE)
        set.seed(42)
        seurat_obj <- RunUMAP(seurat_obj, dims = 1:30, seed.use = 42, verbose = FALSE)
        seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30, verbose = FALSE)
        
        return(seurat_obj)
    }

    # Helper: create a placeholder plot for failed datasets
    make_failed_plot <- function(name) {
        ggplot() +
            annotate("text", x = 0.5, y = 0.5, label = paste0(name, "\n[FAILED]"),
                     size = 5, color = "red", hjust = 0.5) +
            theme_void() +
            ggtitle(name) +
            theme(plot.title = element_text(hjust = 0.5, size = 11, face = "bold", color = "red"))
    }
    
    # Process all datasets (only once)
    seurat_list <- list()
    for (i in seq_along(data_paths)) {
        seurat_list[[i]] <- tryCatch(
            process_one(data_paths[i], data_names[i]),
            error = function(e) {
                cat(paste0("[ERROR] Failed: ", data_names[i], " -- ", conditionMessage(e), "\n"))
                return(NULL)
            }
        )
    }
    
    # Cluster with fixed k using known cell type count, then compute ARI
    all_ari <- data.frame()

    cat(paste0("\n========== Clustering method: ", clustering_method, ", target clusters: ", n_clusters, " ==========\n"))

    for (i in seq_along(seurat_list)) {
        obj <- seurat_list[[i]]

        if (is.null(obj)) {
            all_ari <- rbind(all_ari, data.frame(
                method = data_names[i],
                n_clusters = NA,
                ARI = NA,
                best_res = NA
            ))
            next
        }
        
        if (clustering_method == "kmeans") {
            pca_embed <- Embeddings(obj, reduction = "pca")[, 1:30, drop = FALSE]
            set.seed(42)
            km_res <- kmeans(pca_embed, centers = n_clusters, nstart = 20)
            cluster_labels <- as.character(km_res$cluster)
            obj$self_cluster <- cluster_labels
            selected_res <- NA

        } else if (clustering_method %in% c("louvain", "leiden")) {
            algorithm <- ifelse(clustering_method == "louvain", 1, 4)
            best_res <- 0.8
            best_ari <- -1

            res_grid <- if (clustering_method == "leiden") {
                # seq(0.2, 1.2, by = 0.2)
                seq(0.1, 2.0, by = 0.2)
            } else {
                seq(0.1, 2.0, by = 0.2)
            }   
                     
            for (res in res_grid) {
                obj_temp <- run_find_clusters_safe(
                    obj,
                    resolution = res,
                    algorithm = algorithm,
                    verbose = FALSE
                )
                
                temp_labels <- as.character(obj_temp$seurat_clusters)
                temp_true <- obj_temp$cell_type
                temp_valid <- temp_true != "Unknown"
                
                temp_ari <- adjustedRandIndex(
                    temp_labels[temp_valid],
                    temp_true[temp_valid]
                )
                
                if (!is.na(temp_ari) && temp_ari > best_ari) {
                    best_ari <- temp_ari
                    best_res <- res
                }
                
                rm(obj_temp, temp_labels, temp_true, temp_valid, temp_ari)
                
                if (clustering_method == "leiden") {
                    gc()
                }
            }
            
            obj <- run_find_clusters_safe(
                obj,
                resolution = best_res,
                algorithm = algorithm,
                verbose = FALSE
            )
            
            cluster_labels <- as.character(obj$seurat_clusters)
            obj$self_cluster <- cluster_labels
            selected_res <- best_res

        } else {
            stop("clustering_method must be 'kmeans', 'louvain', or 'leiden'")
        }

        true_labels <- obj$cell_type
        valid_idx <- true_labels != "Unknown"
        if (sum(valid_idx) == 0) {
            ari <- NA
        } else {
            ari <- adjustedRandIndex(cluster_labels[valid_idx], true_labels[valid_idx])
        }
        found_clusters <- length(unique(cluster_labels))

        cat(paste0(
            "  ", data_names[i],
            ": k=", found_clusters,
            ", ARI=", round(ari, 4),
            ", best_res=", selected_res,
            "\n"
        ))

        seurat_list[[i]] <- obj

        all_ari <- rbind(all_ari, data.frame(
            method = data_names[i],
            n_clusters = found_clusters,
            ARI = round(ari, 4),
            best_res = selected_res
        ))

    }

    # Sort ARI and get top methods
    ari_valid <- all_ari[!is.na(all_ari$ARI), ]
    ari_valid <- ari_valid[order(-ari_valid$ARI), ]

    target_prefixes_plot <- c("VAE", "DCA_mse", "scVI_mse", "Transformer_denoise")
    trans_order <- c("no_trans", "count+1", "sqrt", "sqrt+1", "log2", "log2(count+2)")
    other_method_names <- c(
        "Original",
        "Original (Norm 1M)",
        "Original (seurat standard normalize)",
        "scVI",
        "DCA",
        "SAVER"
    )

    get_method_trans <- function(data_name) {
        for (pfx in target_prefixes_plot) {
            if (startsWith(data_name, paste0(pfx, "_"))) {
                suffix <- sub(paste0("^", pfx, "_"), "", data_name)
                norm_patterns <- c("_no_norm$", "_standardize$", "_1000000$", "_100000$", "_10000$", "_1000$")
                trans <- suffix
                for (np in norm_patterns) {
                    if (grepl(np, suffix)) {
                        trans <- sub(np, "", suffix)
                        break
                    }
                }
                return(list(method = pfx, trans = trans))
            }
        }
        return(NULL)
    }

    main_layout <- expand.grid(
        method = target_prefixes_plot,
        trans = trans_order,
        stringsAsFactors = FALSE
    )

    main_idx <- integer(nrow(main_layout))
    for (j in seq_len(nrow(main_layout))) {
        matched_idx <- which(vapply(
            data_names,
            function(x) {
                parsed <- get_method_trans(x)
                !is.null(parsed) &&
                    parsed$method == main_layout$method[j] &&
                    parsed$trans == main_layout$trans[j]
            },
            logical(1)
        ))

        if (length(matched_idx) == 0) {
            main_idx[j] <- NA_integer_
        } else {
            main_idx[j] <- matched_idx[1]
        }
    }

    other_idx <- match(other_method_names, data_names)

    height_umap <- min(length(trans_order) * 5, 48)
    other_width <- width / 4

    # Plot UMAP colored by reference cell type - 6 transformations x 4 methods
    plot_list <- list()
    for (j in seq_along(main_idx)) {
        i <- main_idx[j]
        plot_name <- paste0(main_layout$method[j], "_", main_layout$trans[j])

        if (is.na(i) || is.null(seurat_list[[i]])) {
            plot_list[[length(plot_list) + 1]] <- make_failed_plot(plot_name)
            next
        }

        ari_value <- all_ari$ARI[i]
        ari_title <- ifelse(is.na(ari_value), "ARI = NA", paste0("ARI = ", sprintf("%.3f", ari_value)))

        p <- DimPlot(seurat_list[[i]], reduction = "umap", group.by = "cell_type", 
                    label = TRUE, repel = TRUE, seed = 42) + 
        ggtitle(paste0(data_names[i], "\n", ari_title)) + 
        NoLegend() +
        theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))
        plot_list[[length(plot_list) + 1]] <- p
    }

    combined_plot <- wrap_plots(plot_list, ncol = 4)
    umap_path <- file.path(output_dir, "UMAP_color_by_reference_6x4.png")
    dir.create(dirname(umap_path), showWarnings = FALSE, recursive = TRUE)
    ggsave(umap_path, combined_plot, width = width, height = height_umap, dpi = dpi, limitsize = FALSE)
    cat(paste0("\n[SAVE] UMAP plot: ", umap_path, "\n"))

    # Plot UMAP colored by reference cell type - other methods 6 x 1
    plot_list_other <- list()
    for (j in seq_along(other_idx)) {
        i <- other_idx[j]

        if (is.na(i) || is.null(seurat_list[[i]])) {
            plot_list_other[[length(plot_list_other) + 1]] <- make_failed_plot(other_method_names[j])
            next
        }

        ari_value <- all_ari$ARI[i]
        ari_title <- ifelse(is.na(ari_value), "ARI = NA", paste0("ARI = ", sprintf("%.3f", ari_value)))

        p <- DimPlot(seurat_list[[i]], reduction = "umap", group.by = "cell_type", 
                    label = TRUE, repel = TRUE, seed = 42) + 
        ggtitle(paste0(data_names[i], "\n", ari_title)) + 
        NoLegend() +
        theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))
        plot_list_other[[length(plot_list_other) + 1]] <- p
    }

    combined_plot_other <- wrap_plots(plot_list_other, ncol = 1)
    umap_other_path <- file.path(output_dir, "UMAP_color_by_reference_other_6x1.png")
    dir.create(dirname(umap_other_path), showWarnings = FALSE, recursive = TRUE)
    ggsave(umap_other_path, combined_plot_other, width = other_width, height = height_umap, dpi = dpi, limitsize = FALSE)
    cat(paste0("[SAVE] UMAP plot: ", umap_other_path, "\n"))

    # Save ARI CSV
    csv_path <- file.path(output_dir, paste0("ARI_", clustering_method, ".csv"))
    dir.create(dirname(csv_path), showWarnings = FALSE, recursive = TRUE)
    write.csv(all_ari, csv_path, row.names = FALSE)
    cat(paste0("\n[SAVE] ARI table: ", csv_path, "\n"))

    # Save ARI CSV (best norm per trans for VAE/DCA_mse/scVI_mse/Transformer_denoise)
    target_prefixes <- c("VAE", "DCA_mse", "scVI_mse", "Transformer_denoise")
    best_norm_rows <- list()
    other_rows <- list()
    for (j in seq_len(nrow(all_ari))) {
        m <- all_ari$method[j]
        matched <- FALSE
        for (pfx in target_prefixes) {
            if (startsWith(m, paste0(pfx, "_"))) {
                suffix <- sub(paste0("^", pfx, "_"), "", m)
                norm_patterns <- c("_no_norm$", "_standardize$", "_1000000$", "_100000$", "_10000$", "_1000$")
                trans <- suffix
                for (np in norm_patterns) {
                    if (grepl(np, suffix)) {
                        trans <- sub(np, "", suffix)
                        break
                    }
                }
                key <- paste0(pfx, "|||", trans)
                if (is.null(best_norm_rows[[key]])) {
                    best_norm_rows[[key]] <- j
                } else {
                    prev_ari <- all_ari$ARI[best_norm_rows[[key]]]
                    cur_ari <- all_ari$ARI[j]
                    if (!is.na(cur_ari) && (is.na(prev_ari) || cur_ari > prev_ari)) {
                        best_norm_rows[[key]] <- j
                    }
                }
                matched <- TRUE
                break
            }
        }
        if (!matched) {
            other_rows[[length(other_rows) + 1]] <- j
        }
    }
    keep_idx <- sort(c(unlist(best_norm_rows), unlist(other_rows)))
    all_ari_best_norm <- all_ari[keep_idx, ]
    csv_best_path <- file.path(output_dir, paste0("ARI_", clustering_method, "_best_norm.csv"))
    dir.create(dirname(csv_best_path), showWarnings = FALSE, recursive = TRUE)
    write.csv(all_ari_best_norm, csv_best_path, row.names = FALSE)
    cat(paste0("[SAVE] ARI best norm table: ", csv_best_path, "\n"))

    # Plot ARI bar chart - top 40
    ari_top40 <- head(ari_valid, 40)
    p_ari <- ggplot(ari_top40, aes(x = reorder(method, ARI), y = ARI, fill = method)) +
        geom_col() +
        coord_flip() +
        scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
        labs(x = "Method", y = "ARI", title = paste0("ARI with ", clustering_method, " (target k=", n_clusters, ") - Top 40")) +
        theme_bw() +
        theme(legend.position = "none")

    ari_plot_path <- file.path(output_dir, paste0("ARI_", clustering_method, ".png"))
    dir.create(dirname(ari_plot_path), showWarnings = FALSE, recursive = TRUE)
    ggsave(ari_plot_path, p_ari, width = 12, height = 10, dpi = 300)
    cat(paste0("[SAVE] ARI plot: ", ari_plot_path, "\n"))
    
    return(list(
        plot = combined_plot,
        other_plot = combined_plot_other,
        seurat_list = seurat_list,
        all_ari = all_ari
    ))
}