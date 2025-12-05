#########################################################################
# Helper functions for DDRTree trajectory analysis
#
# Purpose:
# - Configure DDRTree parameters for dimensionality reduction and trajectory learning.
# - Adjust raw clinical/imaging features by removing demographic confounds.
# - Prepare diverse data types (clinical, VAE latents, combined) for Monocle.
#
# Key principle:
# - RAW MEASUREMENTS (e.g., imaging features) are confounded by demographics and
#   NEED covariate adjustment to isolate disease/functional signal.
# - LEARNED REPRESENTATIONS (e.g., VAE latents) are already independent features
#   and generally do NOT need adjustment (they are already decorrelated by training).
#########################################################################

ddrtree_params <- function() {
  ## DDRTree hyperparameters for trajectory learning.
  # `maxIter`: maximum iterations before convergence (higher = more time but better fit).
  # `ncenter`: ~1–5% of sample size; controls MST granularity (more centers = finer tree).
  # `param.gamma`: tradeoff between data fidelity (low) and tree smoothness (high).
  # `tol`: convergence tolerance; lower = stricter but slower.
  # `verbose`: print convergence info during fitting.
  return(data.frame(
    maxIter = 1000, ## 200
    sigma = 0.001,
    ncenter = 550,
    param.gamma = 5,
    tol = 0.001,
    verbose = TRUE  
  ))
}


adjust_covariate <- function(dat) {
  ## Residualize raw clinical/imaging features against demographic confounds.
  # WHY ADJUST:
  # - Raw clinical measurements (e.g., LVEF, LDL) are strongly associated with age/sex/ethnicity.
  # - Residualization isolates disease signal by removing confounding variation.
  # - This allows trajectory analysis to find biological patterns independent of demographics.
  #
  # METHOD: For each feature, fit a polynomial model:
  # y_residual = residuals(lm(y ~ sex + age + age^2 + sex*age + ethnicity))
  # This captures nonlinear and interaction effects.
  require(progress)

  # Extract ID columns and feature columns (exclude metadata).
  eids <- colnames(dat)[startsWith(colnames(dat), "eid_")]
  var_names <- colnames(dat)[!colnames(dat) %in% c(eids, "age_at_MRI", "Sex", "Ethnic_background")]

  pb <- progress::progress_bar$new(total = length(var_names))

  message("Adjust", length(var_names), "variables")

  for (i in seq_along(var_names)) {
    # Fit a polynomial regression model on scaled feature values.
    # Model: y_scaled ~ sex + age + age^2 + sex*age + ethnicity
    model <- lm(y ~ sex + age + age**2 + sex * age + ethn, data = data.frame(
      y = scale(dat[[var_names[i]]]),
      age = dat$age_at_MRI,
      sex = factor(dat$Sex),
      ethn = factor(dat$Ethnic_background)
    ))

    # Return the standardized residuals
    dat[[var_names[i]]] <- c(resid(model))
    pb$tick()
  }

  pb$terminate()

  return(dat)
}



"
Residualizing data on sex and age, or any other covariates, is a common preprocessing step in statistical analysis and machine learning. 
The goal is to remove the effects of these covariates from the variables of interest, 
allowing the analysis to focus on the primary relationships or patterns in the data without the confounding influence of those covariates. 
"
adjust_models <- function(dat, levels_sex, levels_ethn, num_feats_to_adjust = NULL) {
  ## Fit adjustment models (for later application to new data) instead of modifying in-place.
  # WHY SEPARATE:
  # - Allows you to train adjustment models on one set (e.g., healthy controls) and apply
  #   the SAME model to another set, avoiding data leakage and ensuring consistency.
  # - Useful if you later want to adjust test data or new subjects using training set coefficients.
  require(progress)
  
  # Extract ID and feature columns (exclude metadata).
  eids <- colnames(dat)[startsWith(colnames(dat), "eid_")]
  var_names <- colnames(dat)[!colnames(dat) %in% c(eids, "age_at_MRI", "Sex", "Ethnic_background")]
  
  if (!is.null(num_feats_to_adjust)) {
    var_names <- var_names[seq(1, num_feats_to_adjust)]
  }
  
  pb <- progress::progress_bar$new(total = length(var_names))
  
  print("")
  message("Adjust", length(var_names), "variables")
  
  models <- vector("list", length(var_names))
  names(models) <- var_names
  
  for (i in seq_along(var_names)) {
    message(var_names[i])
    # Adjust by polynomial model:
    # Y = a0 + a1 * sex + a2 * age + a3 * age^2 + a4 * sex * age + a5 * ethn
    models[[i]] <- lm(y ~ sex + age + age**2 + sex * age + ethn, data = data.frame(
      y = scale(dat[[var_names[i]]]),
      age = dat$age_at_MRI,
      sex = factor(dat$Sex, levels = levels_sex),
      ethn = factor(dat$Ethnic_background, levels = levels_ethn)
    ))
    pb$tick()
  }
  
  pb$terminate()
  
  return(models)
}


prepare_data_for_model <- function(imaging_data, set_name, adjust = TRUE) {
  ## Prepare data for DDRTree depending on data type and intended analysis.
  print(set_name)
  if (set_name == "CMR") {
    ## Raw cardiac imaging features.
    # WHY ADJUST: Raw LVEF, LV volume, mass, etc. are strong functions of age/sex.
    # Adjust to isolate pathological variation from demographic variation.
    input_data <- imaging_data
    input_data <- adjust_covariate(input_data)
  } else if (set_name == "motion_latent") {
    ## VAE latent embeddings (learned representations of cardiac motion).
    # WHY NO ADJUSTMENT:
    # - VAE latents are already learned, independent features optimized during training.
    # - The VAE encoder learns disentangled factors; latent dims are decorrelated by design.
    # - Residualizing learned features risks removing meaningful variation and hurting downstream analysis.
    # - Use the raw latents directly for trajectory inference.
    print(set_name)
    
    pc_4d <- read_csv("./LVmyo_motion_embeddings_32_VAE_NS_36856.csv", show_col_types = FALSE) %>%
    ## pc_4d <- read_csv("./LVmyo_motion_embeddings_32_VAE_NS_78113.csv", show_col_types = FALSE) %>%
    # pc_4d <- read_csv("./LVmyo_motion_embeddings_32_VAE_NS_41257.csv", show_col_types = FALSE) %>%
      
     

      
      mutate(eid_18545 = as.character(eid_18545)) %>%
      drop_na() 
    
    print(dim(pc_4d))
    input_data <- pc_4d
    print(input_data)
  } else if (set_name == "CMR_motion_latent") {
    ## Combined: raw clinical imaging PLUS motion VAE latents.

    print(set_name)
    # Load motion latent embeddings (historical variants commented below).
    pc_4d <- read_csv("./LVmyo_motion_embeddings_32_VAE_NS_36856.csv", show_col_types = FALSE) %>%   
      mutate(eid_18545 = as.character(eid_18545)) %>%
      drop_na() %>%
      mutate(eid_18545 = as.character(eid_18545))
    print(pc_4d)
    input_data <- inner_join(imaging_data, pc_4d)
    
    # Adjust variables
    input_data <- adjust_covariate(input_data)
    
  } else if (set_name == "Demo_CMR_motion") {
    print(set_name)
    # motion latent ===============================================================
  
    pc_4d <- read_csv("./LVmyo_motion_embeddings_32_VAE_NS_36856.csv", show_col_types = FALSE) %>%
      mutate(eid_18545 = as.character(eid_18545)) %>%
      drop_na() %>%
      mutate(eid_18545 = as.character(eid_18545))
    print(pc_4d)
    
    ## Select specific clinical features for this analysis (subset of full imaging dataset).
    imaging_data_sel = imaging_data %>% select(
      "eid_18545", "age_at_MRI", "Sex", "Ethnic_background",
      "SBP_at_MRI", "DBP_at_MRI", "LVEF", "LVCO", "LVEDVi", "LVESVi", "LVMi",
      "chol_p30690_i0", "HDL_p30760_i0", "LDL_p30780_i0", "HbA1c_p30750_i0"
    ) 

    print(imaging_data_sel)  
    # Adjust clinical features for demographics.
    imaging_data_sel <- adjust_covariate(imaging_data_sel)
    
    print(imaging_data_sel)
    
    input_data <- inner_join(imaging_data_sel, pc_4d)
    
  } else if (set_name == "CMR_ECG_Pheno") {
    ## ECG phenotypic measurements (raw features).
    # WHY ADJUST: Raw ECG measurements (PR interval, QTc, etc.) are age/sex-dependent.
    
    ecg_pheno <- read_csv("/mnt/cardiac/UKBB_40616/ECG/ecg_phenotypes.csv", show_col_types = FALSE) %>%
      mutate(eid_40616 = as.character(eid_40616)) %>%
      select(-PQInterval, -PDuration, -PAxis, -POnset, -POffset) %>% # remove columns with many NAs
      drop_na() %>%
      filter(Instance == 2) %>%
      select(-Instance) %>%
      mutate(eid_40616 = as.character(eid_40616))

    input_data <- inner_join(imaging_data, ecg_pheno)

    # Adjust variables
    input_data <- adjust_covariate(input_data)
  } else if (set_name == "CMR_ECG_Latent") {
    ## ECG latent embeddings (learned from ECG waveforms) + optional clinical features.
    # WHY NO ADJUSTMENT (for ECG latents):
    # - ECG embeddings are learned representations, already independent by design.
    # - If clinical features are included, only adjust those (not the embeddings).
    # ECG latent ===============================================================

    fname_to_eid <- function(x) {
      x <- strsplit(x, "_")[[1]][1]
      x <- as.numeric(x)
      return(x)
    }

    ecg_embedding <- read_csv(here("S:/sk/ECG_embedding.csv")) %>%
      select(-1)

    ecg_fnames <- ecg_embedding %>%
      pull(eid)

    ecg_embedding <- ecg_embedding %>%
      rowwise() %>%
      mutate(eid = fname_to_eid(eid)) %>%
      dplyr::rename(eid_40616 = eid) %>%
      mutate(eid_40616 = as.character(eid_40616))

    is_instance_2 <- lapply(ecg_fnames, function(x) {
      x <- strsplit(x, "_")[[1]][3] %>%
        as.numeric()
      x == 2
    }) %>% unlist()

    ecg_embedding <- ecg_embedding[is_instance_2, ]

    # # Remove outliers =================================
    #
    # any_outlier <- apply(ecg_embedding %>% select(-starts_with("eid")), 2, function(x) {
    #   m1 = median(x)
    #   m2 = mad(x)
    #   (x < m1 - 5 * m2) | (x > m1 + 5 * m2)
    # })
    # is_outlier <- rowSums(any_outlier) > 0
    # ecg_embedding <- ecg_embedding[!is_outlier, ]

    # Conditionally adjust clinical features if requested.
    if (adjust) {
      input_data <- adjust_covariate(imaging_data)
    } else {
      input_data <- imaging_data
    }

    input_data <- input_data %>%
      mutate(eid_40616 = as.character(eid_40616))

    input_data <- inner_join(input_data, ecg_embedding)
  } else {
    stop("Invalid set name")
  }

  return(input_data)
}


# Assigns cells based on the given cell weights.
assign_cells <- function(cell_weights) {
  require(stats)

  if (is.null(dim(cell_weights))) {
    if (any(cell_weights == 0)) {
      stop("Some cells have no positive cell weights.")
    } else {
      return(matrix(1, nrow = length(cell_weights), ncol = 1))
    }
  } else {
    if (any(rowSums(cell_weights) == 0)) {
      stop("Some cells have no positive cell weights.")
    } else {
      # normalize weights
      norm_weights <- sweep(cell_weights, 1,
        FUN = "/",
        STATS = apply(cell_weights, 1, sum)
      )
      # sample weights
      w_samp <- apply(norm_weights, 1, function(prob) {
        stats::rmultinom(n = 1, prob = prob, size = 1)
      })
      # If there is only one lineage, wSamp is a vector so we need to adjust for
      # that
      if (is.null(dim(w_samp))) {
        w_samp <- matrix(w_samp, ncol = 1)
      } else {
        w_samp <- t(w_samp)
      }
      return(w_samp)
    }
  }
}

# This function finds the average node of a given connected component in a graph.
# The average node is considered the "Normal" subject

find_average_node <- function(cds) {
  ## Find the cell closest to the mean of the DDRTree 2D embedding.
  # Useful for selecting a "typical" starting point (root) for pseudotime ordering.
  mean_tree <- apply(cds@reducedDimS, 1, mean)
  nn_res <- nn2(t(cds@reducedDimS), matrix(mean_tree, 1, 2), k = 1)
  return(c(nn_res$nn.idx))
}



## find leaf nodes (endpoints) of the minimum spanning tree.
find_leaf_nodes <- function(mst) {
  ## Identify leaf nodes in the MST (nodes connected to only one other node).
  # Leaf nodes represent endpoints/terminuses of the learned trajectory.
  require(igraph)
  adj_mat <- get.adjacency(mst)
  leaf_nodes <- c()
  for (i in seq_len(nrow(adj_mat))) {
    if (sum(adj_mat[i, ] != 0) == 1) {
      leaf_nodes <- c(leaf_nodes, i)
    }
  }
  return(leaf_nodes)
}




match_dataframe_with_eids <- function(dat, eid, eid_name) {
  ## Match and filter a dataframe by subject IDs using inner join.
  # Keeps only rows whose IDs are present in both `dat` and `eid`.
  require(dplyr)
  
  # Ensure both 'eid' and 'dat[[eid_name]]' are character type for safe matching.
  dat[[eid_name]] <- as.character(dat[[eid_name]])
  eid <- as.character(eid)
  
  # Create a dataframe with the 'eid' values
  eid_df <- data.frame(eid = eid)
  colnames(eid_df) <- eid_name
  
  # Perform an inner join to match IDs
  matched_df <- inner_join(dat, eid_df, by = eid_name)
  
  return(matched_df)
}

match_dataframe_with_eids_left <- function(dat, eid, eid_name) {
  
  # Coerce to character for safe matching
  eid_chr <- as.character(eid)
  dat[[eid_name]] <- as.character(dat[[eid_name]])
  
  # Deduplicate dat on eid_name (keep first row per ID)
  if (anyDuplicated(dat[[eid_name]]) > 0) {
    warning("Duplicate IDs in `dat` for ", eid_name, "; keeping the first occurrence.")
    dat <- dat %>% dplyr::distinct(.data[[eid_name]], .keep_all = TRUE)
  }
  
  # Build an eid frame and LEFT JOIN to preserve order/length of `eid`
  eid_df <- tibble::tibble(!!eid_name := eid_chr)
  out <- dplyr::left_join(eid_df, dat, by = eid_name)
  
  # Optional: warn about unmatched IDs
  n_unmatched <- sum(!eid_chr %in% dat[[eid_name]])
  if (n_unmatched > 0) {
    message(n_unmatched, " ID(s) in `eid` not found in `dat` (rows will contain NAs).")
  }
  
  # Sanity: rows must match length(eid)
  stopifnot(nrow(out) == length(eid_chr))
  
  out
}
