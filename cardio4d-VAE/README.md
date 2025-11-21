# Spatiotemporal Variational Autoencoder for Cardiac Motion (Cardio4D-VAE)


Cardio4D-VAE is a spatiotemporal β-VAE developed to learn compact, generative representations of 4D cardiac motion from dynamic left-ventricular (LV) mesh sequences.  
Trained in an unsupervised setting on UK Biobank participants, the model encodes each cardiac cycle into a $d$-dimensional latent space capturing both anatomical shape and temporal deformation patterns.


---

### 1. Training (`train_vae.py`)
- Train Cardio4D-VAE to learn a probabilistic d-dimensional latent representation.
- The β-VAE objective encourages disentangled and interpretable latent factors.
- Periodically saves model checkpoints and latent embeddings.

### 2. Evaluation (`test_vae.py`)
- Load a trained model and evaluate reconstruction quality.
- Extract per-subject latent vectors $Z_i \in \mathbb{R}^d$.
- Generate reconstructed or synthetic LV motion sequences.

---

## Citation

If you use this repository, please cite the associated publication (to be added).

---

## Contact

For questions or collaborations, please reach out via GitHub issues or email (s.kalaie@imperial.ac.uk).
