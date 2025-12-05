#########################################################################
# Regression models of motion latent representations by phenotype/genotype
#
# Purpose: Perform DDRTree-based lineage analysis and phenotype/regression testing
# Description: Loads a Monocle2-style CellDataSet RDS (`cds_gamma_<gamma>.rds`) produced
#   by DDRTree, aligns sample metadata (phenotypes and genotype labels), computes
#   pseudotime and lineages, fits GAMs  per phenotype and lineage, runs
#   tradeSeq start-vs-end and association tests, and saves plots and summaries under
#   `RESULTS_DIR`.
# Inputs:
#   - `cds_gamma_<gamma>.rds` placed in `RESULTS_DIR` (set via env var `CARDIO_RESULTS_BASE` / `RESULTS_DIR`).
#   - phenotype and metadata CSVs in `INPUT_DIR` / `META_DIR` (overridable via env vars).
# Outputs:
#   - PDF plots and CSV/TXT summaries under subfolders of `RESULTS_DIR` (`phenotype`, `genotype`, etc.).
# Notes:
#   - VAE latent representations are treated as learned features; do NOT residualize them for covariates here.
########################################################################


# install.packages("BiocManager")
# BiocManager::install()
# 
# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
#
# if (!require("monocle", quietly = TRUE))
#   install.packages("monocle")
#
# BiocManager::install("monocle")
#

# Load necessary libraries

library(monocle)
library(DDRTree)
library(parallel)
library(here)
library(glue)
library(dplyr)
library(purrr)
library(ggplot2)
library(mgcv)
library(tradeSeq)
library(tibble)
library(igraph)
library(RANN)
library(VGAM)
library(tidyverse)
library(tidygam)
library(forcats)
library(broom)
library(cowplot)

## Configuration block: set paths and experiment constants here
## You can override these with environment variables if needed.
INPUT_DIR <- Sys.getenv('CARDIO_INPUT_DIR', unset = "s:/sk")
META_DIR  <- Sys.getenv('CARDIO_META_DIR', unset = "p:/motion_code/metadata/HCM_DCM_Rare_Variants/")
# RESULTS_BASE <- Sys.getenv('CARDIO_RESULTS_BASE', unset = INPUT_DIR)
RESULTS_BASE <- getwd()

gamma <- as.integer(Sys.getenv('CARDIO_GAMMA', unset = '5'))
cell_size <- as.numeric(Sys.getenv('CARDIO_CELL_SIZE', unset = '4'))
set.seed(as.integer(Sys.getenv('CARDIO_SEED', unset = '42')))

## Experiment identifiers
set_name <- "motion_latent"
RESULTS_DIR <- file.path(glue(RESULTS_BASE, "./DDRTree_results_40k_{set_name}/ncenter_550"))


## Create the results directories if they do not exist
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

gen_dir <- file.path(RESULTS_DIR, "genotype")
dir.create(gen_dir, showWarnings = FALSE, recursive = TRUE)

# --- ensure results dir exists ---
# disease_dir <- file.path(RESULTS_DIR, "disease_2d")
disease_dir <- file.path(RESULTS_DIR, "disease_1d")
dir.create(disease_dir, showWarnings = FALSE, recursive = TRUE)


pheno_dir <- file.path(RESULTS_DIR, "phenotype")
dir.create(pheno_dir, showWarnings = FALSE, recursive = TRUE)
## FUNCTIONS ===================================================================

source(here("__functions.R"))
## =============================================================================

# RESULTS_DIR <- "cardiac/sk/tree_4D/gamma5"
# INPUT_DIR<- "cardiac/sk"

# RESULTS_DIR <- "./tree_4D"
# INPUT_DIR<- "."

##### Load the cds data

cds_path <- file.path(RESULTS_DIR, paste0("cds_gamma_", gamma, ".rds"))
if (!file.exists(cds_path)) {
  stop("Required file not found: ", cds_path, "\nPlease run DDRTree or point RESULTS_DIR to the folder containing cds_gamma_<gamma>.rds")
}
cds <- readRDS(cds_path)

colnames(cds)
## Metadata =====================================================================


ttn_label<- read_csv(
  file.path(META_DIR,"TTN_full_cohort.csv"),
  show_col_types = FALSE
)%>%
  mutate(eid_40616 = as.character(eid_40616), eid_18545 = as.character(eid_18545))

## DCM PLP variantes
dcm_plp_label<- read_csv(
  file.path(META_DIR,"dcm_plp.csv"),
  show_col_types = FALSE
)%>%
  mutate(eid_40616 = as.character(eid_40616), eid_18545 = as.character(eid_18545))

## HCM PLP variantes
hcm_plp_label<- read_csv(
  file.path(META_DIR,"hcm_plp.csv"),
  show_col_types = FALSE
)%>%
  mutate(eid_40616 = as.character(eid_40616), eid_18545 = as.character(eid_18545))



# Load the phenotype
pheno_49k <- read_csv(
  file.path(INPUT_DIR,"phenotypes_48k_99CMR_13other_all_eid.csv"),
  show_col_types = FALSE
)%>%
  mutate(eid_40616 = as.character(eid_40616), eid_18545 = as.character(eid_18545))


pheno_49k_allmeta <- read_csv(
  file.path(INPUT_DIR,"phenotypes_48k_allmetadata_eid_18545_new.csv"),  ### "phenotypes_48k_allmetadata_eid_18545.csv"
  show_col_types = FALSE
)%>%
  mutate(eid_40616 = as.character(eid_40616), eid_18545 = as.character(eid_18545))



pheno_disease <- read_csv(
  file.path(INPUT_DIR,"final_table_days_from_mri_all_eid.csv"),
  show_col_types = FALSE
)%>%
  mutate(eid_40616 = as.character(eid_40616), eid_18545 = as.character(eid_18545))


pheno_prs <- read_csv(
  file.path(INPUT_DIR,"PRScs_HCM_UKB_eid.csv"),
  show_col_types = FALSE
)%>%
  mutate(eid_40616 = as.character(eid_40616), eid_18545 = as.character(eid_18545), eid_47602 = as.character(eid_47602))

hcm_gen <- read_csv(
  file.path(INPUT_DIR,"HCM_450k_UKBB_genes_thick_thin_alleid.csv"),
  show_col_types = FALSE
)%>%
  mutate(eid_40616 = as.character(eid_40616), eid_18545 = as.character(eid_18545), eid_47602 = as.character(eid_47602))

# Match with the available subjects in the project cohort
eid_cds <- colnames(cds)
pheno_49k <- match_dataframe_with_eids_left(pheno_49k, eid_cds, "eid_18545")
pheno_disease <- match_dataframe_with_eids_left(pheno_disease, eid_cds, "eid_18545")

pheno_49k_allmeta <- match_dataframe_with_eids_left(pheno_49k_allmeta, eid_cds, "eid_18545")
hcm_gen<- match_dataframe_with_eids(hcm_gen, eid_cds, "eid_18545")
# First : Remove subjects that do not have metadata 

mismatched_rows <- !colnames(cds) %in% pheno_49k$eid_18545
# Check if there are any mismatches
if (sum(mismatched_rows) != 0) {
  # Do something when there are mismatches
  print("There are mismatches.")
  
  matched_rows <- colnames(cds) %in% pheno_49k$eid_18545
  cds <- cds[, matched_rows]
  cds@reducedDimS <- cds@reducedDimS[, matched_rows]
} else {
  message("No mismatches found between CDS subjects and metadata.")
}


## Lineages ====================================================================

# Root node
root_node <- find_average_node(cds)
root_node_name <- colnames(cds)[root_node]

# background
outfile <- file.path(RESULTS_DIR, "root_node_tree.png")
png(outfile, width = 1600, height = 1200, res = 300, bg = "white")  # or type="cairo"

plot(cds@reducedDimS[1,], cds@reducedDimS[2,],
     pch=16, col=adjustcolor("grey70", 0.5),
     xlab="Component 1", ylab="Component 2", asp=1)

# map chosen root cells -> MST vertices (ids only)
proj <- cds@auxOrderingData$DDRTree$pr_graph_cell_proj_closest_vertex
root_vertices <- as.integer(proj[root_node, 1])   # root_node can be length k
xy <- t(cds@reducedDimK[, root_vertices, drop=FALSE])

cols <- hcl.colors(length(root_vertices), "Dark 3")
pch  <- rep_len(c(16,17,15,18,3,4,8), length(root_vertices))

points(xy[,1], xy[,2], col=cols, pch=pch, cex=2)
legend("topright", legend=paste("Root", seq_along(root_vertices)),
       col=cols, pch=pch, bty="n")
dev.off()


######################
outfile <- file.path(RESULTS_DIR, "root_node_treebone.png")
png(outfile, width = 1600, height = 1200, res = 300, bg = "white")  # or type="cairo"
plot(cds@reducedDimK[1, ], cds@reducedDimK[2, ])
points(cds@reducedDimK[1, root_vertices],
       cds@reducedDimK[2, root_vertices],
       col = "red"
)
dev.off()
######################
# Minimum spanning tree represents the graph which connects all subjects in the
# tree structure
mst <- cds@minSpanningTree
node_names <- V(mst)$name

ddrtree_graph <- cds@auxOrderingData$DDRTree$pr_graph_cell_proj_tree

# Leaves are the tip nodes, the end point subjects for each trajectory
leaf_nodes <- find_leaf_nodes(mst)
message("Num. leaf states = ", length(leaf_nodes))
## Lineages ====================================================================


# Read the lineage graphs
lineage_graphs<- readRDS(file.path(RESULTS_DIR, "lineages_igraphs.rds"))


## Pseudo time for each lineage ================================================
##Save the data for each lineage for reusing
data_for_models<- readRDS(file.path(RESULTS_DIR, "data_for_models.rds"))
# Generate the pseudo time matrix. This is required for the lineage analysis
all_pseudotime = read.csv(file.path(RESULTS_DIR, "pseudotime_lineages.csv"))

pt_matrix <- matrix(
  NA, ncol(cds), length(leaf_nodes),
  dimnames = list(colnames(cds), paste0("t", seq(1, length(leaf_nodes))))  
)
cell_weights <- (!is.na(all_pseudotime)) * 1


##=================================================================================

## Remove points that are never part of any trajectory

not_isolated <- rowSums(cell_weights) != 0

print(paste('points that are part of any trajectory:', sum(!not_isolated)))
# missing_rows <- which(!not_isolated)
# print(missing_rows)

cds <- cds[, not_isolated]
cds@reducedDimS <- cds@reducedDimS[, not_isolated]
pt_matrix <- pt_matrix[not_isolated, ]
cell_weights <- cell_weights[not_isolated, ]


## ========= Running prediction model for chosen Imaging Traits:
#### # ---- Setup ---------------------------------------------------------------


# Ensure output dirs
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)


# Helper: clean a name for filenames/titles
safe_name <- function(x) {
  gsub("[^A-Za-z0-9_]+", "_", x)
}

# Helper: choose family automatically
choose_family <- function(v) {
  vals <- unique(na.omit(v))
  if (length(vals) <= 2 && all(vals %in% c(0, 1))) {
    print(" binomial is choosen")
    binomial()
  } else {
    print(" gaussian is choosen")
    gaussian()
  }
}

# Helper: median-impute a vector
median_impute <- function(v) {
  v[is.na(v)] <- median(v, na.rm = TRUE)
  v
}

# ---- Core per-phenotype runner ------------------------------------------
run_for_pheno <- function(pheno_name,
                          OUT_DIR,
                          cds,
                          pheno_df,               # pheno_49k_allmeta
                          pt_model_norm,
                          data_matrix_model,
                          lineage_graphs,
                          cell_size = 0.75) {
  
  message(">>> Running: ", pheno_name)
  
  # 1) Prepare response y
  stopifnot(pheno_name %in% colnames(pheno_df))
  y <- median_impute(pheno_df[[pheno_name]])
  # y <- pheno_df[[pheno_name]]
  
  # # Example: enforce a stable order by cell ID
  # ord <- order(colnames(cds))           # or your own stable key
  # y  <- y[ord]
  # data_matrix_model <- data_matrix_model[ord, , drop = FALSE]
  # pt_model_norm     <- pt_model_norm[ord, , drop = FALSE]
  
  print( y)
  # 2) Choose family
  fam <- choose_family(y)
  
  # 3) 2D global GAM on embedding (for coloring the trajectory)
  #    (Adjust these two lines if your reduced dims are stored differently)
  x1 <- cds@reducedDimS[1, ]
  x2 <- cds@reducedDimS[2, ]
  df_global <- tibble(label = y, x1 = x1, x2 = x2)
  
  # Use a mild flexible model for smooth trends over the 2D embedding
  global_mdl <- gam(label ~ x1 + x2 + s(x1, k = 20) + s(x2, k = 20),
                    family = fam, method = "REML", data = df_global)
  
  global_preds <- as.numeric(predict(global_mdl))
  
  # Legend title reflects scale (no "LogOdds" unless binomial)
  leg_title <- if (inherits(fam, "family") && fam$family == "binomial") {
    paste0("Predicted log-odds of ", pheno_name)
  } else {
    paste0("Predicted ", pheno_name)
  }
  
  p_traj <- plot_cell_trajectory(
    cds, color_by = global_preds, cell_size = cell_size,
    show_branch_points = FALSE, show_nodes = FALSE
  ) +
    coord_fixed() +
    scale_color_viridis_c(option = "viridis") +
    labs(color = leg_title, title = paste0("Global GAM: ", pheno_name)) +
    theme(
      legend.key.height = unit(1, "cm"),
      legend.key.width  = unit(2, "cm")
    )
  
  ggsave(
    file.path(OUT_DIR, paste0("plot_", safe_name(pheno_name), "_logOdds_GAM.pdf")),
    width = 15, height = 10, dpi = 300, plot = p_traj, bg = "white"
  )
  
  p_lineage <- plot_df %>%
    ggplot(aes(x = logFC, y = lineage, color = signif)) +
    geom_point() +
    geom_segment(aes(x = 0, xend = logFC, y = lineage, yend = lineage)) +
    scale_y_discrete(limits = plot_df$lineage) +
    geom_vline(xintercept = 0) +
    theme_classic() +
    xlab(paste0("logFC ", pheno_name, " (Path End / Path Start)")) +
    ggtitle(paste0("Start–End test per lineage: ", pheno_name))
  
  
  invisible(list(
    global_model = global_mdl,
    traj_plot_file  = file.path(OUT_DIR, paste0("plot_", safe_name(pheno_name), "_logOdds_GAM.pdf"))
  ))
}

# ---- Run over a vector of phenotypes ------------------------------------


phenos <- c(
  "HbA1c_p30750_i0", "DBP_at_MRI",  "LVEDVi" , "LVESVi","LVMi" , "WT_Global", "circum_PDSR","longit_PDSR"
  # add more, e.g. "DBP", "TTN", "HCM_PLP"
)

all_outputs <- lapply(
  phenos,
  run_for_pheno,
  OUT_DIR = pheno_dir,   # <- pass here
  cds = cds,
  pheno_df = pheno_49k_allmeta,
  pt_model_norm = pt_model_norm,
  data_matrix_model = data_matrix_model,
  lineage_graphs = lineage_graphs,
  cell_size = 4
)




## Find lineages associated to DCM and HCM =====================================

# =====  TTN ==============================
all_ttn <- as.integer(colnames(cds) %in% ttn_label$eid_18545)
print(table(all_ttn))

global_mdl_ttn <- tibble(
  
  label = all_ttn,
  
  x1 = cds @reducedDimS[1, ],
  x2 = cds @reducedDimS[2, ]
) %>%
  {
    # gam(label ~ s(x1, x2, bs = "tp", k = 100),
    #     family = binomial(), method = "REML",
    #     data = ., select = TRUE)
    gam(label ~ x1 + x2 + s(x1, k = 20) + s(x2, k = 20),
        method = "REML",
        family = binomial(), data = .
    )
  }
global_preds_ttn <- predict(global_mdl_ttn )

summary(global_mdl_ttn )

p <- plot_cell_trajectory(cds , color_by = global_preds_ttn,cell_size = cell_size, 
     show_branch_points = FALSE,  # hide numbers
     show_nodes = FALSE           # (optional) also hide the black circles
                          ) + 
  coord_fixed() +
  scale_color_viridis_c(option = "viridis") + 
  labs(color = "LogOdds TTN")+
  theme(
    legend.key.height = unit(1, "cm"),  # Adjust the value as needed
    legend.key.width = unit(2, "cm")    # Adjust the value as needed
    
  )

ggsave(file.path(gen_dir, "plot_TTN_logOdds_GAM.pdf"),
       
       width = 15, height = 10,
       dpi = 300, plot = p, bg = "white"
)

# =====  DCM PLP ==============================
all_dcm_plp <- as.integer(colnames(cds) %in% dcm_plp_label$eid_18545)
print(table(all_dcm_plp))

global_mdl_dcm_plp <- tibble(
  label = all_dcm_plp,
  x1 = cds @reducedDimS[1, ],
  x2 = cds @reducedDimS[2, ]
) %>%
  {
    # gam(label ~ s(x1, x2, bs = "tp", k = 100),
    #     family = binomial(), method = "REML",
    #     data = ., select = TRUE)
    gam(label ~ x1 + x2 + s(x1, k = 20) + s(x2, k = 20),
        method = "REML",
        family = binomial(), data = .
    )
  }
global_preds_dcm_plp <- predict(global_mdl_dcm_plp )

summary(global_mdl_dcm_plp )




p <- plot_cell_trajectory(cds , color_by = global_preds_dcm_plp,cell_size = cell_size, 
  show_branch_points = FALSE,  # hide numbers
  show_nodes = FALSE           # (optional) also hide the black circles                         
  ) + 
  coord_fixed() +
  scale_color_viridis_c(option = "viridis") + 
  labs(color = "LogOdds P/LP DCM variants")+
  theme(
    legend.key.height = unit(1, "cm"),  # Adjust the value as needed
    legend.key.width = unit(2, "cm")    # Adjust the value as needed
    
  )
ggsave(file.path(gen_dir, "plot_DCM_PLP_logOdds_GAM.pdf"),
       
       width = 15, height = 10,
       dpi = 300, plot = p, bg = "white"
)
# =====  HCM PLP ==============================
all_hcm_plp <- as.integer(colnames(cds) %in% hcm_plp_label$eid_18545)
print(table(all_hcm_plp))

global_mdl_hcm_plp <- tibble(
  label = all_hcm_plp,
  x1 = cds @reducedDimS[1, ],
  x2 = cds @reducedDimS[2, ]
) %>%
  {
    # gam(label ~ s(x1, x2, bs = "tp", k = 100),
    #     family = binomial(), method = "REML",
    #     data = ., select = TRUE)
    gam(label ~ x1 + x2 + s(x1, k = 20) + s(x2, k = 20),
        method = "REML",
        family = binomial(), data = .
    )
  }
global_preds_hcm_plp <- predict(global_mdl_hcm_plp )

summary(global_mdl_hcm_plp )

p <- plot_cell_trajectory(cds , color_by = global_preds_hcm_plp,cell_size = cell_size, 
                          show_branch_points = FALSE,  # hide numbers
                          show_nodes = FALSE           # (optional) also hide the black circles                         
) + 
  coord_fixed() +
  scale_color_viridis_c(option = "viridis") + 
  labs(color = "LogOdds P/LP HCM variants")+
  theme(
    legend.key.height = unit(1, "cm"),  # Adjust the value as needed
    legend.key.width = unit(2, "cm")    # Adjust the value as needed
    
  )

ggsave(file.path(gen_dir, "plot_HCM_PLP_logOdds_GAM.pdf"),
       
       width = 15, height = 10,
       dpi = 300, plot = p, bg = "white"
)


safe_name <- function(x) gsub("[^A-Za-z0-9_]+", "_", x)

write_gam_summary <- function(model, label_vec, name, out_dir) {
  s <- summary(model)
  N  <- length(label_vec)
  n1 <- sum(label_vec == 1, na.rm = TRUE)
  n0 <- sum(label_vec == 0, na.rm = TRUE)
  dev_expl <- if (!is.null(s$dev.expl)) round(100 * s$dev.expl, 4) else NA_real_
  
  lines <- capture.output({
    cat("=== ", name, " ===\n", sep = "")
    cat("N:", N, "  Cases:", n1, "  Controls:", n0, "\n")
    cat("Deviance explained (%):", dev_expl, "\n\n")
    print(s)
  })
  fn <- file.path(out_dir, paste0("GAM_", safe_name(name), "_summary.txt"))
  writeLines(lines, fn)
  message("Saved: ", fn)
}

write_gam_summary(global_mdl_ttn, all_ttn, "TTN_variants", gen_dir)
write_gam_summary(global_mdl_dcm_plp, all_dcm_plp, "P-LP_DCM_variants", gen_dir)
write_gam_summary(global_mdl_hcm_plp, all_hcm_plp, "P-LP_HCM_variants", gen_dir)


lo <- min(global_preds_dcm_plp, global_preds_hcm_plp, na.rm = TRUE)
hi <- max(global_preds_dcm_plp, global_preds_hcm_plp, na.rm = TRUE)

logodds_dcm <- global_preds_dcm_plp
logodds_hcm <- global_preds_hcm_plp

p_dcm_scale <- plot_cell_trajectory(cds, color_by = "logodds_dcm",
                              show_branch_points = FALSE, show_nodes = FALSE) +
  coord_fixed() +
  scale_color_viridis_c(limits = c(lo, hi), option = "viridis") +
  labs(color = "LogOdds P/LP DCM")+
  theme(
    legend.key.height = unit(1, "cm"),  # Adjust the value as needed
    legend.key.width = unit(2, "cm")    # Adjust the value as needed
    
  )

p_hcm_scale <- plot_cell_trajectory(cds, color_by = "logodds_hcm",
                              show_branch_points = FALSE, show_nodes = FALSE) +
  coord_fixed() +
  scale_color_viridis_c(limits = c(lo, hi), option = "viridis") +
  labs(color = "LogOdds P/LP HCM")+
  theme(
    legend.key.height = unit(1, "cm"),  # Adjust the value as needed
    legend.key.width = unit(2, "cm")    # Adjust the value as needed
    
  )


diff_logodds <- global_preds_hcm_plp - global_preds_dcm_plp

p_diff <- plot_cell_trajectory(cds, color_by = "diff_logodds",
                     show_branch_points = FALSE, show_nodes = FALSE) +
  coord_fixed() +
  scale_color_gradient2(low = "#3b4cc0", mid = "white", high = "#b40426",
                        midpoint = 0, name = "Diff\nLog-Odds") +
  # labs(title = "Difference map (HCM − DCM)")+
  theme(
    legend.key.height = unit(1, "cm"),  # Adjust the value as needed
    legend.key.width = unit(2, "cm")    # Adjust the value as needed
    
  )


ggsave(file.path(gen_dir, "plot_HCM_PLP_logOdds_GAM_scale.pdf"),
       width = 12, height = 12,
       dpi = 300, plot = p_hcm_scale, bg = "white"
)
ggsave(file.path(gen_dir, "plot_DCM_PLP_logOdds_GAM_scale.pdf"),
       width = 12, height = 12,
       dpi = 300, plot = p_dcm_scale, bg = "white"
)
ggsave(file.path(gen_dir, "plot_diff_PLP_logOdds_GAM_scale.pdf"),
       width = 12, height = 12,
       dpi = 300, plot = p_diff, bg = "white"
)

####. Overly actual DCM/ HCM ========================
# 1) Build 0/1 labels for DCM and HCM aligned to cds columns
# eid_cds <- colnames(cds)
# # ph_d <- match_dataframe_with_eids(pheno_disease, eid_cds, "eid_18545")
# 
# lab01 <- function(x) {
#   xnum <- suppressWarnings(as.numeric(as.character(x)))
#   as.integer(!is.na(xnum) && length(xnum)>0 && xnum != 0)
# }
# 
# dcm <- ifelse(is.na(pheno_disease$DCM), 0L, lab01(pheno_disease$DCM))
# hcm <- ifelse(is.na(pheno_disease$HCM), 0L, lab01(pheno_disease$HCM))

dcm <- as.integer(!is.na(pheno_disease$DCM))
hcm <- as.integer(!is.na(pheno_disease$HCM))


table(dcm)
table(hcm)
# (optional) attach to pData/colData for reference
# pData(cds)$DCM_flag <- dcm
# pData(cds)$HCM_flag <- hcm

# 2) Get cell coordinates from your embedding
coords <- data.frame(
  x = cds@reducedDimS[1, ],
  y = cds@reducedDimS[2, ],
  DCM = factor(dcm, levels = c(0,1), labels = c("False","True")),
  HCM = factor(hcm, levels = c(0,1), labels = c("False","True"))
)

####### DCM =========================


outfile <- file.path(RESULTS_DIR, "genotype", "DCM_tree_ICD10.pdf")
pdf(outfile, width = 6, height = 4.5, pointsize = 8, onefile = FALSE, paper = "special")
# png(outfile, width = 1600, height = 1200, res = 300, bg = "white")
# op <- par(bty="l", mar=c(4,4,1.5,1))
op <- par(bty = "l", mar = c(0, 0, 1.5, 0))   # no box; tiny margins

plot(cds@reducedDimS[1,], cds@reducedDimS[2,],
     pch=16, cex=1.2, col=adjustcolor("grey70",0.5), ## "lightblue2"
     xlab="Component 1", ylab="Component 2", asp=1)

ix <- !is.na(coords$DCM) & coords$DCM == "True"
points(coords$x[ix], coords$y[ix], pch=16, cex=1.2, col=adjustcolor("red3",0.9))
title("DCM")##,cex.main = 0.9)

legend("topright", inset=0.001,
       legend=c("False","True"), pch=16,
       pt.cex=c(1.2,1.2),
       col=c(adjustcolor("grey70",0.5), adjustcolor("red3",0.9))
       # ,title="DCM"
       , bty="n")

par(op); dev.off()
####### HCM =========================
# outfile <- file.path(RESULTS_DIR, "genotype", "HCM_tree.pdf")
# png(outfile, width = 1600, height = 1200, res = 300, bg = "white")

outfile <- file.path(RESULTS_DIR, "genotype", "HCM_tree_ICD10.pdf")
pdf(outfile, width = 6, height = 4.5, pointsize = 8, onefile = FALSE, paper = "special")

# op <- par(bty="l", mar=c(4,4,1.5,1))
op <- par(bty = "l", mar = c(0, 0, 1.5, 0))   # no box; tiny margins

plot(cds@reducedDimS[1,], cds@reducedDimS[2,],
     pch=16, cex=1.2, col=adjustcolor("grey70",0.3),
     xlab="Component 1", ylab="Component 2", asp=1)

ix <- !is.na(coords$HCM) & coords$HCM == "True"
points(coords$x[ix], coords$y[ix], pch=16, cex=1.2, col=adjustcolor("red3",0.9))
title("HCM")

legend("topright", inset=0.001,
       legend=c("False","True"), pch=16,
       pt.cex=c(1.2,1.2),
       col=c(adjustcolor("grey70",0.5), adjustcolor("red3",0.9))
       # ,title="HCM"
       , bty="n")

par(op); dev.off()




#############################################################################################
# ====== Outcomes ======================================================================

# --- helper: safe name for filenames ---
safe_name <- function(x) gsub("[^A-Za-z0-9_]+", "_", x)

# --- helper: align a data.frame to a vector of eids, preserving order/length ---
match_dataframe_with_eids <- function(dat, eid, eid_name) {
  dat[[eid_name]] <- as.character(dat[[eid_name]])
  eid_chr <- as.character(eid)
  # de-duplicate on eid_name (keep first)
  if (anyDuplicated(dat[[eid_name]]) > 0) {
    warning("Duplicate IDs in `dat` for ", eid_name, "; keeping the first occurrence.")
    dat <- dplyr::distinct(dat, .data[[eid_name]], .keep_all = TRUE)
  }
  eid_df <- tibble(!!eid_name := eid_chr)
  out <- dplyr::left_join(eid_df, dat, by = eid_name)
  stopifnot(nrow(out) == length(eid_chr))
  out
}

# --- helper: turn a disease column into a 0/1 label (non-zero/TRUE => 1) ---
to01 <- function(x) {
  # works for logical, numeric, factor, character (e.g., "1","2")
  if (is.logical(x)) return(as.integer(!is.na(x) & x))
  xnum <- suppressWarnings(as.numeric(as.character(x)))
  as.integer(!is.na(xnum) & xnum != 0)
}

# --- data alignment ---
eid_cds <- colnames(cds)  # ensure these are EIDs; if they are cell barcodes, see NOTE below

# --- coordinates from your embedding (Monocle2-style) ---
x1 <- cds@reducedDimS[1, ]
x2 <- cds@reducedDimS[2, ]
stopifnot(length(x1) == ncol(cds), length(x2) == ncol(cds))

# --- diseases to iterate (exclude ID columns) ---
disease_cols <- setdiff(colnames(pheno_disease), c("eid_18545", "eid_40616"))

# --- loop over diseases ---
master_summary <- list()

for (d in disease_cols) {
  message("Processing: ", d)
  lab <- to01(pheno_disease[[d]])
  
  # guardrails
  if (length(lab) != ncol(cds)) {
    warning("Label length mismatch for ", d, " (", length(lab), " vs ", ncol(cds), "). Skipping.")
    next
  }
  n1 <- sum(lab == 1, na.rm = TRUE)
  n0 <- sum(lab == 0, na.rm = TRUE)
  if (n1 == 0 || n0 == 0) {
    # write a brief note and skip model
    txt_path <- file.path(disease_dir, paste0("GAM_", safe_name(d), "_summary.txt"))
    writeLines(c(
      paste0("Disease: ", d),
      paste0("N = ", length(lab), " | cases = ", n1, " | controls = ", n0),
      "Model skipped: outcome has no variation (all 0 or all 1)."
    ), txt_path)
    next
  }
  
  df <- tibble(label = lab, x1 = x1, x2 = x2)
  
  # # fit GAM (2D smooth; shrinkage via select=TRUE)
  # mdl <- gam(label ~ s(x1, x2, bs = "ts", k = 100),
  #            family = binomial(), method = "REML",
  #            data = df, select = TRUE)
  # fit GAM (1D smooth)
  mdl <- gam(label ~ x1 + x2 + s(x1, k = 20) + s(x2, k = 20),
            method = "REML",
            family = binomial(), data = df
        )
  # predictions (log-odds for plotting)
  logodds <- predict(mdl, type = "link")
  # # attach to pData so plot_cell_trajectory can use color_by = "<colname>"
  # # (Monocle2)
  # pData(cds)[[paste0("logodds_", d)]] <- logodds
  # 
  # ---- save text report ----
  s <- summary(mdl)
  dev_expl <- round(100 * s$dev.expl, 4)
  txt_path <- file.path(disease_dir, paste0("GAM_", safe_name(d), "_summary.txt"))
  cap <- capture.output({
    cat("Disease:", d, "\n")
    cat("N:", length(lab), "  Cases:", n1, "  Controls:", n0, "\n")
    cat("Deviance explained (%):", dev_expl, "\n\n")
    print(s)
  })
  writeLines(cap, txt_path)
  
  # keep a row for a master CSV in case you want it later
  master_summary[[d]] <- data.frame(
    disease = d,
    N = length(lab),
    cases = n1,
    controls = n0,
    dev_expl_percent = dev_expl,
    stringsAsFactors = FALSE
  )
  
  # ---- save figure ----
  # color_col <- paste0("logodds_", d)
  color_col <- logodds 
  p <- plot_cell_trajectory(cds, color_by = color_col,
       show_branch_points = FALSE,  # hide numbers
       show_nodes = FALSE           # (optional) also hide the black circles
       ) +
    coord_fixed() +
    scale_color_viridis_c(option = "viridis") +
    labs(color = paste0("LogOdds ", d)) +
    theme(
      legend.key.height = unit(1, "cm"),
      legend.key.width  = unit(2, "cm")
    )
  
  fig_path <- file.path(disease_dir, paste0("plot_", safe_name(d), "_logOdds_GAM.pdf"))
  ggsave(fig_path, width = 12, height = 12, dpi = 300, plot = p, bg = "white")
}

# --- optional: save a master summary table ---
if (length(master_summary)) {
  master_df <- do.call(rbind, master_summary)
  utils::write.table(master_df,
                     file = file.path(disease_dir, "GAM_master_summary.tsv"),
                     sep = "\t", row.names = FALSE, quote = FALSE)
}



#### Outcome correlation with Tree Axis============================

# Use p_r if you truly want correlation coefficients (r).
# Use p_b if you want effect sizes comparable to your screenshot (betas/ORs with CIs).
# ---- axes from your object ----
x1 <- cds@reducedDimS[1, ]
x2 <- cds@reducedDimS[2, ]
stopifnot(length(x1) == ncol(cds), length(x2) == ncol(cds))

# Standardize axes for comparability
sx1 <- as.numeric(scale(x1))
sx2 <- as.numeric(scale(x2))

# Helper: binary 0/1 from various encodings
to01 <- function(x){
  if (is.logical(x)) return(as.integer(!is.na(x) & x))
  xnum <- suppressWarnings(as.numeric(as.character(x)))
  as.integer(!is.na(xnum) & xnum != 0)
}


# Choose LV-mechanics–related diseases (use only those present)
lv_set <- c("MI","IHD","Angina",
            "DCM","HCM",
            "Aortic.Valve.Disease",
            "Amyloidosis",
            "AV.and.conduction.disorders",
            "Myocarditis","Sarcoidosis",
            "Heart.Failure","Hypertension" , "Heart.Failure" , "Stroke", "Atrial.fibrillation.and.flutter" )

disease_cols <- intersect(lv_set, setdiff(colnames(pheno_disease), c("eid_18545","eid_40616")))
stopifnot(length(disease_cols) > 0)

# # ---------- Point-biserial correlation (r) ----------
corr_rows <- list()
for (d in disease_cols){
  y <- to01(pheno_disease[[d]])
  ok <- is.finite(y) & !is.na(y)
  if (sum(y[ok]==1) == 0 || sum(y[ok]==0) == 0) next

  # X axis
  ctX <- suppressWarnings(cor.test(sx1[ok], y[ok]))
  corr_rows[[length(corr_rows)+1]] <- tibble(
    disease = d, axis = "C1", n = sum(ok),
    cases = sum(y[ok]==1), controls = sum(y[ok]==0),
    estimate = unname(ctX$estimate), conf.low = ctX$conf.int[1], conf.high = ctX$conf.int[2],
    p.value = ctX$p.value, metric = "Correlation (r)"
  )
  # Y axis
  ctY <- suppressWarnings(cor.test(sx2[ok], y[ok]))
  corr_rows[[length(corr_rows)+1]] <- tibble(
    disease = d, axis = "C2", n = sum(ok),
    cases = sum(y[ok]==1), controls = sum(y[ok]==0),
    estimate = unname(ctY$estimate), conf.low = ctY$conf.int[1], conf.high = ctY$conf.int[2],
    p.value = ctY$p.value, metric = "Correlation (r)"
  )
}
corr_df <- bind_rows(corr_rows)
corr_df
write.csv(corr_df, file.path(disease_dir, "/new/axes_correlation_r.csv"), row.names = FALSE)

# ---------- Logistic beta per 1 SD (optional) ----------
beta_rows <- list()
for (d in disease_cols){
  y <- to01(pheno_disease[[d]])
  ok <- is.finite(y) & !is.na(y)
  if (sum(y[ok]==1) == 0 || sum(y[ok]==0) == 0) next
  
  # Univariate per-axis logistic (add covariates after ~ if you have them)
  fitX <- glm(y ~ sx1, family = binomial(), subset = ok)   # + age + sex + BSA + HR
  fitY <- glm(y ~ sx2, family = binomial(), subset = ok)
  
  cx <- suppressMessages(confint(fitX))[2,]
  cy <- suppressMessages(confint(fitY))[2,]
  
  beta_rows[[length(beta_rows)+1]] <- tibble(
    disease = d, axis = "C1",
    estimate = coef(fitX)[2], conf.low = cx[1], conf.high = cx[2],
    OR = exp(coef(fitX)[2]), OR.low = exp(cx[1]), OR.high = exp(cx[2]),
    metric = "Log-odds per 1 SD"
  )
  beta_rows[[length(beta_rows)+1]] <- tibble(
    disease = d, axis = "C2",
    estimate = coef(fitY)[2], conf.low = cy[1], conf.high = cy[2],
    OR = exp(coef(fitY)[2]), OR.low = exp(cy[1]), OR.high = exp(cy[2]),
    metric = "Log-odds per 1 SD"
  )
}
beta_df <- bind_rows(beta_rows)
beta_df 
write.csv(beta_df, file.path(disease_dir, "/new/axes_beta_per1SD.csv"), row.names = FALSE)
# ---------- Forest-style plot (Correlation r) ----------
plot_df <- corr_df %>% 
  mutate(disease = fct_reorder(disease, estimate, .fun = median, .desc = TRUE))

p_r <- ggplot(plot_df, aes(x = estimate, y = disease, color = axis)) +
  geom_point(position = position_dodge(width = 0.6), size = 2.6) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                 position = position_dodge(width = 0.6), height = 0) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(x = "Correlation (r)", y = "Incident Cardiovascular Outcomes") + ### "Correlation (r) with axis (z-scored)"
  theme_minimal(base_size = 14)

print(p_r)
ggsave(file.path(disease_dir, "/new/forest_treeaxes_correlation_r.pdf"), p_r, width = 10, height = 6, dpi = 300, bg = "white")

# ---------- Forest-style plot (Log-odds per 1 SD) ----------
plot_beta <- beta_df %>% 
  mutate(disease = fct_reorder(disease, estimate, .fun = median, .desc = TRUE))

p_b <- ggplot(plot_beta, aes(x = estimate, y = disease, color = axis)) +
  geom_point(position = position_dodge(width = 0.6), size = 2.6) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                 position = position_dodge(width = 0.6), height = 0) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(x = "Beta Coefficient", y = "Incident Cardiovascular Outcomes") + ## "Beta (log-odds) per 1 SD of axis"
  theme_minimal(base_size = 14)

print(p_b)
ggsave(file.path(disease_dir, "new/forest_treeaxes_betacoeffLog-odds.pdf"), p_b, width = 10, height = 6, dpi = 300, bg = "white")



##################################################################################

# Fit model

for (i in seq_len(length(leaf_nodes))) {
  message("Path: ", i)
  match_idx <- match(data_for_models[[i]]$id, rownames(pt_matrix))
  pt_matrix[match_idx, i] <- data_for_models[[i]]$pseudotime
  cell_weights[match_idx, i] <- 1
}



w_samp <- assign_cells(cell_weights = cell_weights)
colnames(w_samp) <- paste0("l", seq_len(ncol(w_samp)))

pt_model <- pt_matrix
pt_model[is.na(pt_model)] <- 0

pt_model_norm <- pt_model
for (i in seq_len(ncol(pt_model_norm))) {
  pt_model_norm[, i] <- (pt_model_norm[, i] - min(pt_model_norm[, i])) /
    (max(pt_model_norm[, i]) - min(pt_model_norm[, i]))
}

data_matrix_model <- cbind(data.frame(pt_model_norm), data.frame(w_samp))


## Start-End differential analysis =============================================

smooth_form <- as.formula(
  paste0(
    "y ~ ",
    paste(
      vapply(seq_len(ncol(pt_model_norm)), function(ii) {
        paste0("s(t", ii, ", by=l", ii, ", bs = 'cr', k=6)")
      }, FUN.VALUE = "formula"),
      collapse = "+"
    )
  )
)

# ####=========================================================================================
# #### Linear Regression:
# ####=========================================================================================

##====  if it is Binary values ======

gam_list <- list(
  mgcv::bam(smooth_form,
            data = cbind(data_matrix_model, data.frame(y = all_hcm_plp )),  # all_dcm_plp, all_ttn, all_hyperten
            family = binomial("logit"),
            control = mgcv::gam.control(), method = "REML"
  )
)

##====  if it is Continuous values ======
pheno_49k_allmeta$HbA1c_p30750_i0 <- ifelse(is.na(pheno_49k_allmeta$HbA1c_p30750_i0), median(pheno_49k_allmeta$HbA1c_p30750_i0, na.rm = TRUE), pheno_49k_allmeta$HbA1c_p30750_i0)
all_HbA = (pheno_49k_allmeta$HbA1c_p30750_i0)

gam_list <- list(
  mgcv::bam(smooth_form,
            data = cbind(data_matrix_model, data.frame(y = all_HbA)),  # all_HbA  all_DBP
            family =gaussian(),
            control = mgcv::gam.control(), method = "REML"
  )
)



# logfc <- tradeSeq::startVsEndTest(ngam_list, global = FALSE, lineages = TRUE) %>%   
logfc <- tradeSeq::startVsEndTest(gam_list, global = FALSE, lineages = TRUE) %>%
  t()
# results <- tradeSeq::associatiocnTest(ngam_list, global = FALSE, lineages = TRUE)
print(logfc)
results <- tradeSeq::associationTest(gam_list, global = FALSE, lineages = TRUE)
results <- matrix(results[, -ncol(results)], length(lineage_graphs), 3, byrow = TRUE)
# results <- matrix(results[, -ncol(results)], 10, 3, byrow = TRUE)

colnames(results) <- c("waldStat", "df", "p_value")

results <- cbind(results, data.frame(logFC = logfc[-1, ]))

results$p_BH <- p.adjust(results$p_value, "BH")

print(results)



plot_df <- results %>%
  mutate(lineage = paste0("Lineage", seq_len(nrow(results)))) %>%
  arrange(logFC) %>%
  mutate(signif = p_BH < 0.05)
p <- plot_df %>%
  ggplot(aes(x = logFC, y = lineage, color = signif)) +
  geom_point() +
  geom_segment(aes(
    x = logFC - logFC, xend = logFC, y = lineage,
    yend = lineage
  )) +
  scale_y_discrete(limits = plot_df$lineage) +
  geom_vline(xintercept = 0) +
  theme_classic() +
  xlab("logFC HbA1c_p30750_i0 Path End / Path Start")
# xlab("logFC DBP Path End / Path Start")
  # xlab("logFC TTN Path End / Path Start")
# xlab("logFC HCM_PLP Path End / Path Start")
# +ggtitle(paste0("Lineage", lin))

# ggsave(file.path(RESULTS_DIR, "start_end_test_dcm.png"),
# ggsave(file.path(RESULTS_DIR, "start_end_test_hcm.png"),
# ggsave(file.path(RESULTS_DIR, "start_end_stroke.png"),
# ggsave(file.path(RESULTS_DIR, "start_end_hypertension.png"),
# ggsave(file.path(RESULTS_DIR, "start_end_dcm3hets.png"),
# ggsave(file.path(RESULTS_DIR, "start_end_ttn_.pdf"),
# ggsave(file.path(RESULTS_DIR, "start_end_hcm_plp_.pdf"),
# ggsave(file.path(RESULTS_DIR, "start_end_dcm_plp.png"),
ggsave(file.path(RESULTS_DIR, "start_end_HbA_new.pdf"),
# ggsave(file.path(RESULTS_DIR, "start_end_DBP_.pdf"),
       
       width = 6, height = 6,
       dpi = 300, plot = p, bg = "white"
)



## sanity check:
## Use the lineage model, not the global x1/x2 model
fit <- gam_list[[1]]  # <- THIS is the mgcv::bam(smooth_form, ...)

## Build a base newdata with the same predictor columns the model expects
resp <- all.vars(formula(fit))[1]
pred_vars <- setdiff(names(fit$model), resp)   # should be c(t1..tK, l1..lK)

base <- as.data.frame(matrix(0, nrow = 2, ncol = length(pred_vars)))
names(base) <- pred_vars

## Choose lineage and endpoints in pseudotime
lin <- 6
t0 <- 0.05; t1 <- 0.95  # evaluate near the ends (more stable than 0/1)

## Set the lineage indicator and its pseudotime for the two rows
base[ , paste0("l", lin)] <- 1
base[1, paste0("t", lin)] <- t0
base[2, paste0("t", lin)] <- t1

## Predict from the lineage model
preds <- as.numeric(predict(fit, newdata = base, type = "response"))
delta <- preds[2] - preds[1]   # End - Start

cat("Lineage", lin, "Δ (end - start) =", delta, "\n")
if (delta > 0) cat("=> logFC should be POSITIVE (higher at end)\n") else
  cat("=> logFC should be NEGATIVE (lower at end)\n")



# dat is what you used to fit: dat <- cbind(data_matrix_model, data.frame(y = y))
dat <- cbind(data_matrix_model, data.frame(y = all_HbA))
fhat <- as.numeric(predict(fit, type = "response"))
idx  <- dat[[paste0("l", lin)]] == 1
cor_t <- cor(dat[[paste0("t", lin)]][idx], fhat[idx], use = "complete.obs")
cat("Cor(pred, t", lin, ") =", round(cor_t, 3), "\n")
# cor_t > 0 ⇒ increases along pseudotime (positive); cor_t < 0 ⇒ decreases
####################################

## ===============  Lineage association with Phenotype prediction on the Tree 
# Vector of phenotypes
phenos <- c(
  "HbA1c_p30750_i0", "DBP_at_MRI", "LVEDVi", "LVESVi",
  "LVMi", "WT_Global", "circum_PDSR", "longit_PDSR"
)

# Loop through each phenotype
for (phe in phenos) {
  message(">>> Running phenotype: ", phe)
  
  # Impute missing with median
  pheno_49k_allmeta[[phe]] <- ifelse(
    is.na(pheno_49k_allmeta[[phe]]),
    median(pheno_49k_allmeta[[phe]], na.rm = TRUE),
    pheno_49k_allmeta[[phe]]
  )
  all_y <- pheno_49k_allmeta[[phe]]
  
  # Fit GAM
  gam_list <- list(
    mgcv::bam(
      smooth_form,
      data = cbind(data_matrix_model, data.frame(y = all_y)),
      family = gaussian(),
      control = mgcv::gam.control(),
      method = "REML"
    )
  )
  
  # Run start–end test
  logfc <- tradeSeq::startVsEndTest(gam_list, global = FALSE, lineages = TRUE) %>%
    t()
  
  results <- tradeSeq::associationTest(gam_list, global = FALSE, lineages = TRUE)
  results <- matrix(results[, -ncol(results)], length(lineage_graphs), 3, byrow = TRUE)
  colnames(results) <- c("waldStat", "df", "p_value")
  
  results <- cbind(results, data.frame(logFC = logfc[-1, ]))
  results$p_BH <- p.adjust(results$p_value, "BH")
  
  print(results)
  

  
  # Make plot
  plot_df <- results %>%
    rownames_to_column(var = "lineage") %>% 
    as.data.frame() %>%
    mutate(lineage = paste0("Lineage", seq_len(nrow(results)))) %>%
    arrange(logFC) %>%
    mutate(signif = p_BH < 0.05)
  
  print(plot_df)
  
  # Save numeric results 
  out_csv <- file.path(pheno_dir, paste0("start_end_results_", phe, ".csv")) 
  write.csv(plot_df, out_csv, row.names = FALSE)
  
  
  p <- ggplot(plot_df, aes(x = logFC, y = lineage, color = signif)) +
    geom_point() +
    geom_segment(aes(x = 0, xend = logFC, yend = lineage)) +
    scale_y_discrete(limits = plot_df$lineage) +
    geom_vline(xintercept = 0) +
    theme_classic() +
    xlab(paste0("logFC ", phe, " Path End / Path Start"))
  
  
  ggsave(
    file.path(pheno_dir, paste0("start_end_", phe, ".pdf")),
    width = 6, height = 6, dpi = 300, plot = p, bg = "white"
  )
}


### For Outcome/disease ########################################################################
# ---- diseases to iterate (exclude ID columns) ----
disease_cols <- setdiff(colnames(pheno_disease), c("eid_18545", "eid_40616"))

# --- helper: turn a disease column into a 0/1 label (non-zero/TRUE => 1) ---
to01 <- function(x) {
  # works for logical, numeric, factor, character (e.g., "1","2")
  if (is.logical(x)) return(as.integer(!is.na(x) & x))
  xnum <- suppressWarnings(as.numeric(as.character(x)))
  as.integer(!is.na(xnum) & xnum != 0)
}

# ---- helper: run one disease label through the pipeline ----
run_one_disease <- function(dname) {
  print(dname)
  # y <- pheno_disease[[dname]]
  y<- to01(pheno_disease[[dname]])
  
  # print(y)
  # # coerce to 0/1 integer
  # if (is.logical(y)) y <- as.integer(y)
  # if (is.factor(y))  y <- as.integer(y) - 1L
  # y <- as.integer(y)
  
  # build modeling frame (assumes row order aligns with data_matrix_model)
  df <- cbind(data_matrix_model, y = y)
  df <- df[complete.cases(df), , drop = FALSE]
  if (length(unique(df$y)) < 2L) {
    message("Skipping ", dname, ": no variation in label.")
    return(tibble())
  }
  
  # fit GAM on pseudotime basis terms (your smooth_form)
  fit <- mgcv::bam(
    smooth_form,
    data = df,
    family = binomial("logit"),
    method = "REML",
    control = mgcv::gam.control()
  )
  
  glist <- list(fit)
  
  # tradeSeq tests
  logfc_mat <- tradeSeq::startVsEndTest(glist, global = FALSE, lineages = TRUE)
  logfc     <- t(logfc_mat)              # transpose to vector
  at        <- tradeSeq::associationTest(glist, global = FALSE, lineages = TRUE)
  
  # reshape associationTest to lineage rows (matches your earlier code)
  k <- length(lineage_graphs)            # number of lineages
  mat <- matrix(at[, -ncol(at)], k, 3, byrow = TRUE)
  colnames(mat) <- c("waldStat", "df", "p_value")
  
  
  # results tibble
  results <- as_tibble(mat) |>
    mutate(
      lineage = seq_len(n()),
      logFC   = as.numeric(logfc[-1, ]), # drop intercept row
      disease = dname,
      p_BH       = p.adjust(p_value, method = "BH")  # FDR per disease
    )
  
  # ---- PLOT: identical to your reference style ----
  plot_df <- results %>%
    mutate(lineage = paste0("Lineage", seq_len(nrow(results)))) %>%
    arrange(logFC) %>%
    mutate(signif = p_BH < 0.05)
  
  p <- plot_df %>%
    ggplot(aes(x = logFC, y = lineage, color = signif)) +
    geom_point() +
    geom_segment(aes(
      x = logFC - logFC, xend = logFC, y = lineage,
      yend = lineage
    )) +
    scale_y_discrete(limits = plot_df$lineage) +
    geom_vline(xintercept = 0) +
    theme_classic() +
    xlab("logFC Path End / Path Start")
  
  # ggsave(file.path(disease_dir, paste0("start_end_", dname, ".pdf")),
  #        width = 6, height = 12, dpi = 300, plot = p, bg = "white")
  
  results
}


# ---- run all diseases ----

# all_results <- map_dfr("Hypertension", run_one_disease)

all_results <- map_dfr(disease_cols, run_one_disease)

# Optional: global FDR across ALL lineage×disease tests
all_results <- all_results |>
  mutate(q_global = p.adjust(p_value, method = "BH"))


print(all_results)
# Save table
readr::write_csv(all_results, file.path(disease_dir, "start_end_all_diseases.csv"))

## Association Variables Lineages ==============================================
## Question : Does latent feature z_k increase or decrease from the start to the end of a lineage??

### For each latent feature and each lineage, compares the predicted expression at the start of the lineage vs the end.
## We use a GAM model to predict each latent feature’s value at the beginning vs. the end of each lineage
## 1. Fit a smooth function of pseudotime, stratified by lineage.
## 2. This produces a model of how that latent feature evolves along pseudotime.



# Set the number of features to test
print(featureNames(cds))
# Monocle/ExpressionSet-style
featureNames(cds) <- sub("^c_", "z_", featureNames(cds))

# sanity check
head(featureNames(cds))
stopifnot(!anyDuplicated(featureNames(cds)))

print(featureNames(cds))


num_feats_to_test <- length(featureNames(cds))

# Define a function to fit a GAM to the data
fit_gam <- function(y, pt_model_norm, data_matrix_model) {
  # Construct a formula for the smooth function
  smooth_formula <- as.formula(
    paste0(
      "y ~ ",
      paste(
        # Create a term for each column in the model
        vapply(seq_len(ncol(pt_model_norm)), function(ii) {
          paste0("t", ii, " + s(t", ii, ", by=l", ii, ", bs='cr', k=6)")
        }, FUN.VALUE = "formula"),
        collapse = "+"
      )
    )
  )
  
  # Fit a Generalized Additive Model (GAM) to the data
  mgcv::bam(smooth_formula,
            data = cbind(data_matrix_model, data.frame(y = y)),
            control = mgcv::gam.control(),
            select = TRUE,
            REML = TRUE
  )
}

# Fit a GAM to each feature
cl <- parallel::makeCluster(parallel::detectCores() - 1)
clusterExport(cl, varlist = c("pt_model_norm", "data_matrix_model", "fit_gam"))

data_partitions <- lapply(seq_len(num_feats_to_test), function(i) {
  cds@assayData$exprs[i, ]
})

gam_list <- parallel::parLapply(cl, X = data_partitions, fun = function(x) {
  fit_gam(x, pt_model_norm, data_matrix_model)
})

stopCluster(cl)

names(gam_list) <- rownames(cds)[seq_len(num_feats_to_test)]


print(gam_list)
# saveRDS(
#   gam_list, file.path(RESULTS_DIR, "vars_lineage_association_gam_list.rds")
# )
gam_listaaaaaaaa <- readRDS(file.path(RESULTS_DIR, "vars_lineage_association_gam_list.rds"))

## Compare Start vs End points in all lineages =================================

## Start vs End Test Execution

results <- tradeSeq::startVsEndTest(gam_list,
                                    global = FALSE, lineages = TRUE,
                                    l2fc = 0
)
results <- as.data.frame(results)
# adjust p-values
p_adj <- p.adjust(c(results %>% select(starts_with("pvalue_"))) %>% unlist(),
                  method = "BH"
)
p_adj <- matrix(p_adj, nrow(results), (length(p_adj) / nrow(results)))
colnames(p_adj) <- paste0("padj_lineage", seq_len(ncol(p_adj)))
results <- cbind(results, as.data.frame(p_adj))

# Set same domain for plotting
max_lfc <- -Inf
min_lfc <- Inf
for (lin in seq_len(length(lineage_graphs))) {
  tmp <- results %>% pull(!!sym(paste0("logFClineage", lin)))
  if (min(tmp) < min_lfc) {
    min_lfc <- min(tmp)
  }
  if (max(tmp) > max_lfc) {
    max_lfc <- max(tmp)
  }
  rm(tmp)
}



for (lin in seq_len(length(lineage_graphs))) {
  plot_df <- results %>%
    filter(!!sym(paste0("padj_lineage", lin)) < 0.05) %>%
    arrange(!!sym(paste0("logFClineage", lin))) %>%
    select(!!sym(paste0("logFClineage", lin))) %>%
    rownames_to_column("variable") %>%
    dplyr::rename(logFC = !!sym(paste0("logFClineage", lin)))
  
  p <- ggplot(plot_df, aes(x = logFC, y = variable, color = logFC)) +
    geom_segment(aes(x = 0, xend = logFC, yend = variable), size = 1.2) +  # colored line
    geom_point(size = 3) +  # colored point
    scale_y_discrete(limits = plot_df$variable) +
    scale_x_continuous(limits = c(min_lfc, max_lfc)) +
    geom_vline(xintercept = 0, color = "black") +
    scale_color_viridis_c(option = "viridis") +
    theme_classic() +
    xlab("logFC Path End / Path Start") +
    ggtitle(paste0("Lineage ", lin))
  
  ggsave(
    filename = file.path(
      RESULTS_DIR, "start_end_test_lineage", paste0("lineage_", lin, "_colored.pdf")
    ),
    width = 6, height = 12, dpi = 300, plot = p, bg = "white"
  )
}
# write.csv(results, file = file.path(RESULTS_DIR, "start_end_test_lineages.csv"))



