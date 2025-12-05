#########################################################################
# Tree Lineage Analysis of Motion Latent Representations
#
# Purpose: construct DDRTree lineages from a Monocle2 `cds` object, compute
# pseudotime per lineage, fit GAMs (mgcv) and run tradeSeq start-vs-end /
# association tests, and produce plots/reports under `RESULTS_DIR`.
#
# Notes:
# - Configure `INPUT_DIR`, `RESULTS_DIR`, `gamma`, etc. in the config block below
#   or via environment variables in a wrapper script.
# - This file is an exploratory top-level analysis script (monolithic). For
#   production use consider refactoring into functions and a CLI.
#########################################################################


# Required packages (keep this list minimal and non-duplicated)
library(monocle)
library(DDRTree)
library(parallel)
library(here)
library(glue)
library(igraph)
library(RANN)
library(VGAM)
library(mgcv)
library(tidyverse)
library(tidygam)
library(reshape2)
library(ggplot2)
library(pheatmap)
library(tradeSeq) 


## ---------------------- Configuration ----------------------
# Set the directory paths and experiment constants here. Override via
# environment variables from a wrapper script for reproducible runs.
# Example: Sys.setenv(CARDIO_INPUT_DIR = "s:/sk")
INPUT_DIR <- getwd()
gamma <- 5

set_name = "motion_latent"
RESULTS_DIR <- file.path(glue(INPUT_DIR,"/DDRTree_results_40k_{set_name}/ncenter_550"))

## Ensure results directory exists (create recursively)
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
## FUNCTIONS ===================================================================

source(here("__functions.R"))
## =============================================================================

##### Load the cds data
cds <- readRDS(file.path(RESULTS_DIR,paste0("cds_gamma_", gamma, ".rds")))

## Lineages ====================================================================

# Lineages represent the shortest paths from the center of the tree to the
# tips of each branch. The tip is defined as the node with only one neighbor.
# Nodes in chains have two neighbors, nodes in bifurcations have 3 neighbors.

## Select root node: find the cell closest to the tree center (central pseudotime)
root_node <- find_average_node(cds)
root_node_name <- colnames(cds)[root_node]


# Minimum spanning tree represents the graph which connects all subjects in the
# tree structure
mst <- cds@minSpanningTree
node_names <- V(mst)$name

ddrtree_graph <- cds@auxOrderingData$DDRTree$pr_graph_cell_proj_tree

# Leaves are the tip nodes, the end point subjects for each trajectory
leaf_nodes <- find_leaf_nodes(mst)
message("Num. leaf states = ", length(leaf_nodes))




# Map root cells to closest MST vertices
root_vertices <- cds@auxOrderingData$DDRTree$pr_graph_cell_proj_closest_vertex[root_node, ]
root_vertices <- as.vector(root_vertices)
k <- length(root_vertices)

# (Optional) draw the background first
plot(cds@reducedDimS[1, ], cds@reducedDimS[2, ],
     pch = 16, col = adjustcolor("grey70", 0.5),
     xlab = "Component 1", ylab = "Component 2", asp = 1)

# Colors / shapes
cols <- hcl.colors(k, "Dark 3")
pch  <- rep_len(c(16, 17, 15, 18, 3, 4, 8), k)

# Plot every root
for (i in seq_len(k)) {
  text(cds@reducedDimK[1, root_vertices[i]],
         cds@reducedDimK[2, root_vertices[i]],
         col = cols[i], pch = pch[i], cex = 2)
}

legend("topright",
       legend = paste0("Root ", seq_len(k)),
       col = cols, pch = pch, bty = "n")

# Map root nodes to MST centroids
root_nodes_in_mst <- cds@auxOrderingData$DDRTree$pr_graph_cell_proj_closest_vertex[root_node, ]




# Plot every root
for (i in seq_len(k)) {
  points(cds@reducedDimK[1, root_nodes_in_mst[i]],
         cds@reducedDimK[2, root_nodes_in_mst[i]],
         col = cols[i], pch = pch[i], cex = 2)
  # (Optional) label them
  # text(cds@reducedDimK[1, root_vertices[i]],
  #      cds@reducedDimK[2, root_vertices[i]],
  #      labels = i, pos = 3, cex = 0.8)
}

legend("topright",
       legend = paste0("Root ", seq_len(k)),
       col = cols, pch = pch, bty = "n")


# Plot first root in red
points(cds@reducedDimK[1, root_nodes_in_mst[10]],
       cds@reducedDimK[2, root_nodes_in_mst[10]],
       col = "red", pch = 19, cex = 2)
# If there is a second root, plot it in blue
if (length(root_nodes_in_mst) > 1) {
  points(cds@reducedDimK[1, root_nodes_in_mst[2]],
         cds@reducedDimK[2, root_nodes_in_mst[2]],
         col = "blue", pch = 17, cex = 2)
}

legend("topright",
       legend = c("Root 1", "Root 2"),
       col = c("red", "blue"), pch = c(19, 17))
# Shortest Paths Calculation ===================================================

# Remember: shortest paths are lineages.
# For each branch take the point which has the largest geodesic distance from the root node

root_node_in_mst <-
  cds@auxOrderingData$DDRTree$pr_graph_cell_proj_closest_vertex[root_node, ]
# Visualize the position of the reference node
plot(cds@reducedDimK[1, ], cds@reducedDimK[2, ])
points(cds@reducedDimK[1, root_node_in_mst[1]],
       cds@reducedDimK[2, root_node_in_mst[1]],
       col = "red"
)

# Find the shortest paths from the root node to the leaf nodes
paths_to_leaf <- vector("list", length(leaf_nodes))
for (i in seq_len(length(leaf_nodes))) {
  paths_to_leaf[[i]] <- shortest_paths(
    mst,
    from = node_names[root_node_in_mst[1]], to = node_names[leaf_nodes[i]]
  )
}

# For each path take the closest points
closest_points_to_mst_nodes <- function(cds, mst, mst_node_name) {
  rownames(cds@auxOrderingData$DDRTree$pr_graph_cell_proj_closest_vertex)[
    cds@auxOrderingData$DDRTree$pr_graph_cell_proj_closest_vertex[, 1] ==
      as.numeric(substr(mst_node_name, 3, nchar(mst_node_name)))
  ]
}

lineage_graphs <- vector("list", length(paths_to_leaf))
points_shortest_path <- vector("list", length(paths_to_leaf))

for (ii in seq_along(paths_to_leaf)) {

  message(ii, "/", length(paths_to_leaf))
  mst_nodes_in_path <- node_names[as.numeric(paths_to_leaf[[ii]]$vpath[[1]])]
  points_in_path <- vector("list", length(paths_to_leaf[[ii]]$vpath[[1]]))
  for (kk in seq_along(mst_nodes_in_path)) {
    points_in_path[[mst_nodes_in_path[kk]]] <- closest_points_to_mst_nodes(
      cds, mst, mst_nodes_in_path[kk]
    )
  }
  points_in_path <- unlist(points_in_path) %>% unique()
  points_idx <- sapply(points_in_path, function(x) {
    which(V(ddrtree_graph)$name == x)
  }) %>% unlist()
  stopifnot(length(points_in_path) == length(points_idx))
  lineage_graphs[[ii]] <- subgraph(ddrtree_graph, points_idx)
  
  root_node_index <- which(V(ddrtree_graph)$name == root_node_name)
  if (root_node_index %in% points_idx) {
    print("Root node is included in the points_idx.")
  } else {
    print("Root node is NOT included in the points_idx.")
  }
  
  # Check that the root node is in the sub graph
  stopifnot(root_node_name %in% V(lineage_graphs[[ii]])$name) ####
  
  # Some sub graphs are disconnected. This is a problem if we want to find the
  # shortest path from the root node to the edge of the tree.
  # To solve this issue, we connect the disconnected sub graphs by adding an
  # edge between the closest points of the closest sub graphs.
  # We start from the component which contains the root node, and determine the
  # farthest point. This becomes the anchor for this component.
  # Then we find the other component which has the closest node to the anchor,
  # and we connect these two points.
  # The process is repeated until we have only 1 connected component.
  
  # Determine if it's disconnected, in case take the component connected to the
  # root node
  conn_comp <- components(lineage_graphs[[ii]])
  while (conn_comp$no != 1) {
    message("num. connected components = ", conn_comp$no)
    
    root_component <- conn_comp$membership[
      names(conn_comp$membership) == root_node_name
    ]
    
    # Root component
    tmp_subgraph <- subgraph(
      lineage_graphs[[ii]],
      which(conn_comp$membership == root_component)
    )
    root_node_in_subgraph <- which(V(tmp_subgraph)$name == root_node_name)
    dist_from_root <- distances(tmp_subgraph, root_node_in_subgraph)
    farthest_point <- V(tmp_subgraph)$name[which.max(dist_from_root)]
    
    # Find the connected component with the closest point to the farthest point
    # from the root (root component)
    other_components <- unique(conn_comp$membership)[
      !unique(conn_comp$membership) %in% root_component
    ]
    min_dist_from_root_component <- c()
    closest_point_root_comp <- c()
    for (j in seq_along(other_components)) {
      other_component <- other_components[j]
      tmp_subgraph <- subgraph(
        lineage_graphs[[ii]],
        which(conn_comp$membership == other_component)
      )
      
      col_idx_farthest_point <- which(colnames(cds) == farthest_point)
      points_in_other_subgraph <- V(tmp_subgraph)$name
      dist_from_farthest_point <- sapply(points_in_other_subgraph, function(x) {
        sum(
          (cds@reducedDimS[, which(colnames(cds) == x)] -
             cds@reducedDimS[, col_idx_farthest_point])**2
        )
      })
      
      min_dist_from_root_component <- c(
        min_dist_from_root_component,
        min(dist_from_farthest_point)
      )
      closest_point_root_comp <- c(
        closest_point_root_comp,
        points_in_other_subgraph[which.min(dist_from_farthest_point)]
      )
    }
    
    # Select component to merge
    sel_comp <- other_component[which.min(min_dist_from_root_component)]
    closest_point <- closest_point_root_comp[
      which.min(min_dist_from_root_component)
    ]
    
    # Connect the two points
    nodes_to_connect <- c(
      which(V(lineage_graphs[[ii]])$name == farthest_point),
      which(V(lineage_graphs[[ii]])$name == closest_point)
    )
    lineage_graphs[[ii]] <- add_edges(lineage_graphs[[ii]],
                                      nodes_to_connect,
                                      weight = 1
    )
    
    # Recalculate the number of connected components
    conn_comp <- components(lineage_graphs[[ii]])
  }
  
  root_node_in_subgraph <- which(V(lineage_graphs[[ii]])$name == root_node_name)
  dist_from_root <- distances(lineage_graphs[[ii]], v = root_node_in_subgraph)
  
  sorted_nodes_by_dist <- order(-dist_from_root)
  points_shortest_path[[ii]] <- shortest_paths(
    lineage_graphs[[ii]],
    from = root_node_in_subgraph,
    to = which.max(dist_from_root), mode = "all"
  )
  conn_comp <- components(lineage_graphs[[ii]])
  message("num. connected components = ", conn_comp$no)
}

# Save the lineage graphs
saveRDS(lineage_graphs, file = file.path(RESULTS_DIR, "lineages_igraphs.rds"))

# lineage_graphs<- readRDS(file.path(RESULTS_DIR, "lineages_igraphs.rds"))


## Pseudo time for each lineage ================================================

data_for_models <- vector("list", length(lineage_graphs))
plots_lineages <- vector("list", length(lineage_graphs))


for (i in seq_along(lineage_graphs)) {
  message(i, "/", length(lineage_graphs))
  
  points_in_path <- as_ids(points_shortest_path[[i]]$vpath[[1]])
  sel_columns <- sapply(points_in_path, function(x) which(colnames(cds) == x))
  # Check if sel_columns is empty or invalid
  if (length(sel_columns) == 0) {
    message("No valid columns found for lineage ", i)
    next
  }
  
  # Ensure sel_columns is a numeric vector, not a list
  if (is.list(sel_columns)) {
    sel_columns <- unlist(sel_columns)
  }
  
  cds_new <- cds
  cds_new <- cds_new[, sel_columns]
  cds_new@auxOrderingData$DDRTree$pr_graph_cell_proj_tree <- lineage_graphs[[i]]
  cds_new@reducedDimS <- cds_new@reducedDimS[, sel_columns]
  cds_new$State <- cds_new$State[sel_columns]
  
## For each lineage: compute pseudotime using Monocle's orderCells()
  cds_new <- orderCells(cds_new)
  
  # Invert pseudo time if necessary
  if (cds_new$Pseudotime[1] >
      cds_new$Pseudotime[length(sel_columns)]) {
      # cds_new$Pseudotime[length(cds_new$Pseudotime)]) {
    cds_new$Pseudotime <- max(cds_new$Pseudotime) - cds_new$Pseudotime
  }
  
  plots_lineages[[i]] <- plot_cell_trajectory(cds_new,
                                              color_by = cds_new$Pseudotime
  ) +
    coord_fixed() +
    scale_color_viridis_c(option = "viridis") +   ###magma
    ggtitle(paste0("Lineage", i))+
    theme(plot.title = element_text(size = 20))  # Adjust title font size
  
  
  data_for_models[[i]] <- tibble(
    id = colnames(cds_new),
    pseudotime = cds_new$Pseudotime
  )
  
  rm(cds_new)
}



# Plot all lineages colored by their pseudo time
p_combo <- cowplot::plot_grid(
  plotlist = plots_lineages, ncol = 3, nrow = 3,
  align = "hv"
)
ggsave(file.path(RESULTS_DIR, "lineages.png"),
       width = 30, height = 20,
       dpi = 300, plot = p_combo, bg = "white"
)

for (i in seq_along(lineage_graphs)){
  ggsave(file.path(RESULTS_DIR, paste0("lineages_",i,".png")),
         width = 10, height = 15,
         dpi = 300, plot = plots_lineages[[i]], bg = "white"
  )
}


# Save the data for each lineage for reusing
saveRDS(data_for_models, file.path(RESULTS_DIR, "data_for_models.rds"))
# data_for_models<- readRDS(file.path(RESULTS_DIR, "data_for_models.rds"))


##=================================================================================

## Build pseudotime matrix: row=sample, col=lineage; value=pseudotime (NA if not in lineage)

all_pseudotime <- matrix(
  NA, ncol(cds), length(data_for_models),
  dimnames = list(colnames(cds), paste0("t", seq(1, length(data_for_models))))
)
for (i in seq_len(length(data_for_models))) {
  tb1 <- tibble(SID = colnames(cds))
  tb2 <- tibble(
    SID = data_for_models[[i]]$id,
    pseudotime = data_for_models[[i]]$pseudotime
  )
  tb_merge <- left_join(tb1, tb2)
  
  all_pseudotime[, i] <- tb_merge$pseudotime
}

write.csv(as.data.frame(all_pseudotime),
          file = file.path(RESULTS_DIR, "pseudotime_lineages.csv")
)


# all_pseudotime = read.csv(file.path(RESULTS_DIR, "pseudotime_lineages.csv"))


