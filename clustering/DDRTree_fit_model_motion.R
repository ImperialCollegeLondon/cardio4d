#########################################################################
# Fit DDRTree model to motion latent data
#
# Purpose:
# - Load precomputed imaging / motion features, prepare them for Monocle,
#   run DDRTree dimensionality reduction (trajectory learning), and save
#   reduced coordinates, state assignments and plots for different gamma
#   parameter values.
#
# Usage:
# - Place this script in a working directory containing the input CSVs
#   and `__functions.R` (provides helper functions such as
#   `ddrtree_params()` and `prepare_data_for_model()`).
# - Set `INPUT_DIR` or run from the project root (default: `getwd()`).
# - Results are written under `RESULTS_DIR` (created automatically).
#########################################################################

## Required libraries -------------------------------------------------------
# `monocle` for DDRTree and trajectory tools, `magrittr` for pipes,

# install.packages("BiocManager")
# BiocManager::install()
# 
# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# 
# if (!require("monocle", quietly = TRUE))
#   install.packages("monocle")
# 
# BiocManager::install("monocle")

library(monocle)
library(magrittr)
library(here)
library(glue)
library(tidyverse)


## Analysis configuration --------------------------------------------------
# `SECTION` describes the anatomical region / feature group used.
SECTION <- 'LVmyo'



## By default use current working directory; override if needed.
INPUT_DIR <- getwd()


## Source helper functions --------------------------------------------------
# This file must define `ddrtree_params()` and `prepare_data_for_model()`
## (and any other helper used below).
source("__functions.R")


### Data loading & preprocessing --------------------------------------------
## Begin analysis

##Load the imputed imaging features
imaging_data <- read_csv(here(file.path(
    INPUT_DIR,
    "img_features_all_rf_imputed_with_biochemistry.csv"
  ))
)




## Ensure ID column is character and categorical covariates are factors.
imaging_data <- imaging_data %>%
  mutate(
    #eid_40616 = as.character(eid_40616),
    eid_18545 = as.character(eid_18545),
    Sex = as.factor(Sex),
    Ethnic_background = as.factor(Ethnic_background)
  )


print(sum(is.na(imaging_data)))

# DDRTree ======================================================================

# Load the ddrtree_params function
param <- ddrtree_params()

## Name used for output folder (keeps results organized)
set_name <- "motion_latent"


# Create the output directory
RESULTS_DIR <- file.path(glue(INPUT_DIR,"/DDRTree_results_40k_{set_name}/ncenter_{param$ncenter}"))

## Create results directory (including parents) if it doesn't exist
dir.create(RESULTS_DIR, recursive = TRUE)

# Prepare the input data for the model
input_data <- prepare_data_for_model(imaging_data, set_name = set_name)


## Convert selected features to a numeric matrix for Monocle.
# We drop ID columns (starting with "eid") and keep only numeric features.
mat_values <- input_data %>%
  select(-starts_with("eid")) %>%
  as.matrix()


# print(mat_values)
# Create the observation metadata
obs_metadata <- data.frame(
  eid_18545 = input_data$eid_18545
) %>%
  set_rownames(as.character(input_data$eid_18545))

# Create the feature metadata
feat_metadata <- data.frame(vert = colnames(mat_values)) %>%
  set_rownames(colnames(mat_values))

## Create Monocle CellDataSet. We transpose and z-score features across
## samples (rows become features). `uninormal()` is appropriate for
## continuous, approximately Gaussian latent features.
cds <- monocle::newCellDataSet(
  t(scale(mat_values)),
  phenoData = AnnotatedDataFrame(obs_metadata),
  featureData = AnnotatedDataFrame(feat_metadata),
  expressionFamily = uninormal()
)



eid_cds <- colnames(cds)
pheno_40k <- match_dataframe_with_eids(imaging_data, eid_cds, "eid_18545")


## Run DDRTree for several `param.gamma` values to compare topology/scale.
for (param.gamma in c(2,5,10)) {


  set.seed(123456)

  
  ## Reduce dimensionality with DDRTree. Important parameters:
  # - `scaling`: whether Monocle rescales features internally (we set FALSE
  #   because we already scaled the matrix above with `scale()`).
  # - `param.gamma`: controls DDRTree tradeoff between data fit and tree smoothness.
  # - `ncenter`: number of centers used by DDRTree (controls granularity).
  cds <- reduceDimension(cds,
    reduction_method = "DDRTree", norm_method = "none",
    pseudo_expr = 0, scaling = FALSE, verbose = TRUE,
    relative_expr = FALSE, maxIter = param$maxIter,
    tol = param$tol, param.gamma = param.gamma,
    ncenter = param$ncenter
  )

  # Create a tibble with reduced dimensions and observation IDs
  tibble(
    c1 = reducedDimS(cds)[1, ],
    c2 = reducedDimS(cds)[2, ],
    eid_18545 = colnames(cds)
  ) %>%
    write_csv(
      file.path(
        RESULTS_DIR,
        glue("ddrtree_gamma_{param.gamma}_ncenter_{param$ncenter}.csv")
      )
    )

  ## Order cells along the learned trajectory and extract state labels.
  cds <- monocle::orderCells(cds)
  state <- cds$State

  # Save the CellDataSet object
  saveRDS(cds, file.path(RESULTS_DIR, glue("cds_gamma_{param.gamma}.rds")))
  saveRDS(state, file.path(RESULTS_DIR, glue("state_gamma_{param.gamma}.rds")))

  # Create trajectory plots colored by pseudotime and by discrete state.
  plot6 = plot_cell_trajectory(cds, color_by = "Pseudotime",cell_size = 2)
  plot8 = plot_cell_trajectory(cds, color_by = 'state',cell_size = 2)

    # Save plots (PDF and SVG) for high-quality figures.
    ggsave(file.path(RESULTS_DIR, glue("Pseudotime_gamma_{param.gamma}.pdf")),
      width = 15, height = 10,
      dpi = 300, plot = plot6, bg = "white")


  ggsave(file.path(RESULTS_DIR, glue("state_gamma_{param.gamma}.pdf")),
         width = 15, height = 10,
         dpi = 300, plot = plot8, bg = "white")

  ggsave(file.path(RESULTS_DIR, glue("state_gamma_{param.gamma}.svg")),
         width = 15, height = 10,
         dpi = 300, plot = plot8, bg = "white")

    ggsave(file.path(RESULTS_DIR, glue("Pseudotime_gamma_{param.gamma}.svg")),
      width = 15, height = 10,
      dpi = 300, plot = plot6, bg = "white")
 

}

#
# #=====================================================================================================

