import pandas as pd
import os
import seaborn as sns
import numpy as np
import math
from tqdm import tqdm
import argparse
import pyvista as pv
import torch
import torch.nn as nn
import pickle

from torch.utils.data import Dataset, DataLoader

from utils import MotionDataset
from utils import compute_total_correlation, beta_ae_lossT, beta_tcvae_lossT, beta_vae_lossT, beta_vae_loss
# from utils import save_checkpoint, plot_training_curves, compute_latent_correlation, adaptive_beta, save_loss_plot, plot_training_curves_ae
# from utils import reverse_preprocessing, save_as_vtk, save_as_vtk_emb, save_as_vtk_emb_smooth
from utils import save_checkpoint, compute_latent_correlation, adaptive_beta, save_loss_plot, plot_training_curves_ae,plot_training_curves_ae_all, plot_training_curves_all
from utils import reverse_preprocessing, save_as_vtk, save_as_vtk_emb, save_as_vtk_emb_smooth
from model import AE, Cardio4DVAE
import json


def train_AE_smooth(model, modelName, train_loader, val_loader, optimizer, output_dir, n_frames, n_nodes, args):
    
    """
    Train the autoencoder with validation and save validation reconstructions as VTK files.
    """

    
    # Store values for visualization
    alpha_values = []
    gamma_values = []
    train_losses, val_losses = [], []
    train_norm_losses,  train_recon_losses, train_lap_losses = [], [], []
    val_norm_losses,  val_recon_losses, val_lap_losses = [], [], []
    model.train()
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    ##============================================================
    # Start with low values for alpha and beta
    alpha = args.alpha
    beta = args.beta
    # beta = 0
    gamma = args.gamma
    

    
    ##============================================================
    
    for epoch in range(args.epochs):
        model.train()
        # Training phase
        total_train_loss = 0.0
        total_train_recon_loss = 0.0
        total_train_lap_loss = 0.0
        total_train_norm_loss = 0.0
        ## Gradually increase alpha and beta
        ## alpha = min(alpha_max, (epoch / total_epochs) * alpha_max)  # Linear warm-up for alpha
        ## beta = min(beta_max, epoch * beta_step)  # Gradual increase of beta
        
        # alpha = min(1.0, epoch * 0.01)  # Increase gradually
        # beta = min(3e-5, epoch * 0.001)  # Increase gradually
        
        
        # alpha_start = 0.1  # Start from 0.1 instead of 0
        # beta_start = 1e-6  # Start from a lower beta

        # alpha = min(1.0, alpha_start * (1.1 ** epoch))  # Exponential increase
        # beta = min(3e-5, beta_start * (1.05 ** epoch))  # Exponential increase
        
        
        # Dynamic Beta
        # beta = adaptive_beta(epoch, start_beta=2e-5, max_beta=args.beta, warmup_epochs=15)
        # beta = adaptive_beta(epoch, start_beta=1e-6, max_beta=args.beta, warmup_epochs=15)

        
        for inputs, _, _, _ in train_loader:
            inputs = inputs.to(device)  ## input size: [batch_size, C, T, H, W]  torch.Size([8, 3, 50, 37, 38])
            # print("input", inputs.shape)
            # encoded, decoded = model(inputs)
            encoded, decoded = model(inputs)
            # print("decoded", decoded.shape)  ## decoded torch.Size([8, 3, 50, 37, 38])
            # loss = criterion(decoded.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1))  # Flatten inputs
            # loss, recon_loss, kl_loss = beta_vae_loss(decoded, inputs, mu, log_var, beta)
            loss, recon_loss,  lap_loss, norm_loss = beta_ae_lossT(decoded, inputs, n_nodes, alpha, beta,  gamma, args.atlas, args) 
            # loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss = beta_tcvae_lossT(decoded, inputs, mu, log_var, z, n_nodes, alpha, beta, args.gamma, args.delta, args.atlas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_recon_loss += recon_loss.item()
            total_train_lap_loss += lap_loss.item()
            total_train_norm_loss += norm_loss.item()

        ## Step the scheduler
        scheduler.step()
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_recon_loss = 0.0
        total_val_lap_loss = 0.0
        total_val_norm_loss = 0.0
        total_val_sparse_loss = 0.0

        embeddings_list = []  # Collect embeddings here
        subject_ids_list = []  # Collect subject IDs
        with torch.no_grad():
            for inputs, subject_ids, original_mins, original_maxs in val_loader:
                inputs = inputs.to(device)
                # encoded, decoded = model(inputs)
                encoded, decoded = model(inputs)
                
                
                ## Check if z is Actually Different for Different Inputs
                # print(f"Sample {subject_ids}: Mean of `mu`: {mu.mean().item():.6f}, Std: {mu.std().item():.6f}")
                
                # loss = criterion(decoded.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1)) 
                # loss, recon_loss, kl_loss = beta_vae_loss(decoded, inputs, mu, log_var, beta)
                loss, recon_loss,  lap_loss, norm_loss = beta_ae_lossT(decoded, inputs,  n_nodes, alpha, beta, gamma, args.atlas, args)
                # loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss = beta_tcvae_lossT(decoded, inputs, mu, log_var, z, n_nodes, alpha, beta, args.gamma, args.delta, args.atlas)
                total_val_loss += loss.item()
                total_val_recon_loss += recon_loss.item()
                total_val_lap_loss += lap_loss.item()
                total_val_norm_loss += norm_loss.item()
                
                
                # Save embeddings and subject IDs
                # embeddings_list.append(encoded.detach().cpu())
                embeddings_list.append(encoded.detach().cpu())
                
                subject_ids_list.extend(subject_ids)  # Extend to handle batch size > 1
                for i, subject_id in enumerate(subject_ids):
                    original_shape = inputs[i].shape  # Shape of the original data   original: torch.Size([3, 50, 37, 38])
                    # print("original:", inputs[i].shape )
                    reconstructed = reverse_preprocessing(
                        decoded[i],
                        n_nodes,
                        original_shape,
                        original_mins[i],
                        original_maxs[i],
                        normalize=True
                    )  ## reconstructed torch.Size([50, 1406, 3])
                    # print("reconstructed", reconstructed.shape)
                    
                    original = inputs[i].permute(1, 2, 3, 0).view(n_frames, -1, 3).cpu()  # Restore [T, Nodes, C] ## reconstructed torch.Size([50, 1406, 3])
                    original = original[:, :n_nodes, :]        # Remove padding (if any)
                    original = original * (original_maxs[i] - original_mins[i]) + original_mins[i]
                    # print("original", original.shape)
                    
                    # original = inputs[i].cpu().numpy()
                    save_as_vtk(reconstructed.cpu() , original, subject_id, output_dir, args)

        # # Combine embeddings and save to CSV
        # embeddings = torch.vstack(embeddings_list).numpy()  # Stack collected tensors
        # id_df = pd.DataFrame(subject_ids_list, columns=["eid_18545"])
        # embeddings_df = pd.DataFrame(embeddings, columns=[f"c_{i+1}" for i in range(embeddings.shape[1])])

        # # Combine IDs with embeddings
        # combined_df = pd.concat([id_df, embeddings_df], axis=1)

        # # Save to CSV
        # embeddings_file = os.path.join(output_dir, f"validation_embeddings_epoch_{epoch+1}.csv")
        # combined_df.to_csv(embeddings_file, index=False)
        # print(f"Embeddings saved to {embeddings_file}")


        # Print epoch losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_train_recon_loss = total_train_recon_loss / len(train_loader)
        avg_train_lap_loss = total_train_lap_loss / len(train_loader)
        avg_val_recon_loss = total_val_recon_loss / len(val_loader)
        avg_val_lap_loss = total_val_lap_loss / len(val_loader)
        avg_train_norm_loss = total_train_norm_loss / len(train_loader)
        avg_val_norm_loss = total_val_norm_loss / len(val_loader)

        alpha_values.append(alpha)
        gamma_values.append(gamma)
        
        # Append losses to lists
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_recon_losses.append(avg_train_recon_loss)
        train_lap_losses.append(avg_train_lap_loss)
        train_norm_losses.append(avg_train_norm_loss)
        val_recon_losses.append(avg_val_recon_loss)
        val_lap_losses.append(avg_val_lap_loss)
        val_norm_losses.append(avg_val_norm_loss)
        
          
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        compute_latent_correlation(output_dir, f"latent_corr_epoch_{epoch+1}", embeddings_list)

        ### ===========================================================================
        ### Save weight ===============================================================

        save_checkpoint(output_dir, modelName, model, optimizer, train_losses, val_losses, args.epochs)
        # Save plot every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            # save_loss_plot(train_losses, val_losses, output_dir, epoch + 1, name="Total_loss")    
            # save_loss_plot(train_recon_losses, val_recon_losses, output_dir, epoch + 1, name="Rec_loss")   
            # save_loss_plot(train_kl_losses, val_kl_losses, output_dir, epoch + 1, name="KL_loss")    
            plot_training_curves_ae(alpha_values, gamma_values, train_losses, val_losses, train_recon_losses, val_recon_losses, train_lap_losses, val_lap_losses, output_dir)

            save_checkpoint(output_dir, f"{modelName}_epoch_{epoch+1}", model, optimizer, train_losses, val_losses, args.epochs)

    print('==============')
    print('Well Done!')
    print('==============')
    # return val_losses , val_recon_losses



    #     print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
    #     ### ===========================================================================
    #     ### Save weight ===============================================================
    #     compute_latent_correlation(output_dir, f"latent_corr_epoch_{epoch+1}", embeddings_list)
    #     # save_checkpoint(output_dir, modelName, model, optimizer, train_losses, val_losses, args.epochs)
    #     save_checkpoint(output_dir, f"{modelName}_epoch_{epoch+1}", model, optimizer, train_losses, val_losses, args.epochs)

    #     # Save plot every 10 epochs
    #     if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
    #         # save_loss_plot(train_losses, val_losses, output_dir, epoch + 1, name="Total_loss")    
    #         # save_loss_plot(train_recon_losses, val_recon_losses, output_dir, epoch + 1, name="Rec_loss")   
    #         # save_loss_plot(train_kl_losses, val_kl_losses, output_dir, epoch + 1, name="KL_loss")    
    #         plot_training_curves(alpha_values, gamma_values, train_losses, val_losses, train_recon_losses, val_recon_losses, train_kl_losses, val_kl_losses, train_lap_losses, val_lap_losses, output_dir)

    #         save_checkpoint(output_dir, f"{modelName}_epoch_{epoch+1}", model, optimizer, train_losses, val_losses, args.epochs)
    # print('==============')
    # print('Well Done!')
    # print('==============')




# def test_model_and_save_embeddings(model, model_path, test_loader, output_name, output_dir, args, rec):
#     """
#     Load a trained model, run it on test data, and save embeddings to a CSV file.
#     If 'rec' is True, save reconstructed meshes, compute Chamfer Distance, and
#     save Chamfer losses per subject to a CSV file.
#     """

#     import os
#     import torch
#     import pandas as pd
#     from tqdm import tqdm
#     import torch.nn as nn
#     from pytorch3d.loss import chamfer_distance as chamfer_dis
    
#     criterion = nn.MSELoss()


#     # Load model
#     print("Loading the trained model...")

#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#     print("Model loaded successfully!")

#     # Initialize containers
#     embeddings_list = []
#     subject_ids_list = []
#     losses = []  # To store dicts of {subject_id, loss}

#     print("Processing test data and saving embeddings...")

#     with torch.no_grad():
#         with tqdm(total=len(test_loader), desc="Processing Batches", unit="batch") as pbar:
#             for inputs, subject_ids, original_mins, original_maxs in test_loader:
#                 inputs = inputs.to(device)
#                 mu, log_var, z, decoded = model(inputs)
                
#                 embeddings_list.append(mu.detach().cpu())
#                 subject_ids_list.extend(subject_ids)

#                 if rec:
#                     print('Reconstruction...')
#                     for i, subject_id in enumerate(subject_ids):
#                         original_shape = inputs[i].shape  # [3, 50, 37, 38]

#                         reconstructed = reverse_preprocessing(
#                             decoded[i],
#                             n_nodes,
#                             original_shape,
#                             original_mins[i],
#                             original_maxs[i],
#                             normalize=True
#                         )  # [50, 1406, 3]

#                         # Restore original
#                         original = inputs[i].permute(1, 2, 3, 0).view(n_frames, -1, 3)
#                         original = original[:, :n_nodes, :]
#                         original = original * (original_maxs[i] - original_mins[i]) + original_mins[i]

#                         #### Save VTKs
#                         save_as_vtk(reconstructed.cpu(), original.cpu(), subject_id, output_dir, args)

#                         # Compute Chamfer Distance
#                         reconstructed = reconstructed.to(device)
#                         original = original.to(device)
#                         dist, _ = chamfer_dis(reconstructed.float(), original.float())
#                         # loss_value = dist.item()
#                         chamfer_value = dist.item()

#                         original_flat = original.view(-1, 3)
#                         reconstructed_flat = reconstructed.view(-1, 3)
#                         mse_value = criterion(reconstructed_flat, original_flat).item()
#                         rmse_value = (torch.sqrt(criterion(reconstructed_flat, original_flat))).item()
                        
                        
                        
#                         # print(f"{subject_id} - Average Chamfer Distance across the frames: {loss_value:.6f}")
#                         # losses.append({"eid_18545": subject_id, "chamfer_distance": loss_value})
#                         print(f"{subject_id} - Chamfer: {chamfer_value:.6f}, MSE: {mse_value:.6f}, RMSE: {rmse_value:.6f}")
#                         losses.append({
#                             "eid_18545": subject_id,
#                             "chamfer_distance": chamfer_value,
#                             "mse_loss": mse_value,
#                             "rms_distance": rmse_value
#                         })

#                 pbar.update(1)

#     # Report average Chamfer Distance
#     if rec and losses:
#         avg_chamfer = sum([d["chamfer_distance"] for d in losses]) / len(losses)
#         print(f"\n Average Chamfer Distance across test set: {avg_chamfer:.6f}")


#         avg_mse = sum([d["mse_loss"] for d in losses]) / len(losses)
#         print(f"\n Average MSE Distance across test set: {avg_mse:.6f}")
        
        
#         avg_rmse = sum([d["rms_distance"] for d in losses]) / len(losses)
#         print(f"\n Average RMSE Distance across test set: {avg_rmse:.6f}")
        
        

#         # Save Chamfer distances to CSV
#         chamfer_df = pd.DataFrame(losses)
#         chamfer_csv_path = os.path.join(output_dir, f"losses_test_{len(test_loader)}.csv")
#         chamfer_df.to_csv(chamfer_csv_path, index=False)
#         print(f"Distances saved to {chamfer_csv_path}")

#     # Save embeddings to CSV
#     embeddings = torch.vstack(embeddings_list).numpy()
#     id_df = pd.DataFrame(subject_ids_list, columns=["eid_18545"])
#     embeddings_df = pd.DataFrame(embeddings, columns=[f"c_{i+1}" for i in range(embeddings.shape[1])])
#     combined_df = pd.concat([id_df, embeddings_df], axis=1)

#     embeddings_file = os.path.join(output_dir, output_name)
#     combined_df.to_csv(embeddings_file, index=False)
#     print(f"Embeddings saved to {embeddings_file}")

def test_model_and_save_embeddings(model, model_path, test_loader, output_name, output_dir, args, rec):
    """
    Load a trained model, run it on test data, and save embeddings to a CSV file.
    If 'rec' is True, save reconstructed meshes, compute Chamfer Distance, and
    save Chamfer losses per subject to a CSV file.
    
    
    
    inputs: torch.Size([1, 3, 50, 30, 40])
    decoded: torch.Size([1, 3, 50, 30, 40])
    Reconstruction...
    original: torch.Size([50, 1200, 3])
    reconstructed: torch.Size([50, 1200, 3])
    """
    

    import os
    import torch
    import pandas as pd
    from tqdm import tqdm
    import torch.nn as nn
    from pytorch3d.loss import chamfer_distance as chamfer_dis
    
    criterion = nn.MSELoss()


    # Load model
    print("Loading the trained model...")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Initialize containers
    embeddings_list = []
    subject_ids_list = []
    losses = []  # To store dicts of {subject_id, loss}

    print("Processing test data and saving embeddings...")

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Processing Batches", unit="batch") as pbar:
            for inputs, subject_ids, original_mins, original_maxs in test_loader:
                inputs = inputs.to(device)
                # print("inputs:", inputs.shape)
                mu, log_var, z, decoded = model(inputs)
                # print("decoded:", decoded.shape)
                embeddings_list.append(mu.detach().cpu())
                subject_ids_list.extend(subject_ids)

                if rec:
                    print('Reconstruction...')
                    for i, subject_id in enumerate(subject_ids):
                        original_shape = inputs[i].shape  # [3, 50, 37, 38]

                        reconstructed = reverse_preprocessing(
                            decoded[i],
                            n_nodes,
                            original_shape,
                            original_mins[i],
                            original_maxs[i],
                            normalize=True ## took shape to the original space
                        )  # [50, 1406, 3]

                        # Restore original
                        original = inputs[i].permute(1, 2, 3, 0).view(n_frames, -1, 3)
                        original = original[:, :n_nodes, :]
                        original = original * (original_maxs[i] - original_mins[i]) + original_mins[i]

                        # print("original:", original.shape)
                        # print("reconstructed:", reconstructed.shape)
                        #### Save VTKs
                        # save_as_vtk(reconstructed.cpu(), original.cpu(), subject_id, output_dir, args)

                        ############# Computing losses in original space #############################################################
                        # # Compute Chamfer Distance
                        reconstructed = reconstructed.to(device)
                        original = original.to(device)
                        dist, _ = chamfer_dis(reconstructed.float(), original.float())
                        # loss_value = dist.item()
                        chamfer_value = dist.item()

                        # original_flat = original.view(-1, 3)
                        # reconstructed_flat = reconstructed.view(-1, 3)
                        original_flat = original.reshape(1, -1)
                        reconstructed_flat = reconstructed.reshape(1, -1)
                        mse_value = criterion(reconstructed_flat, original_flat).item()
                        rmse_value = (torch.sqrt(criterion(reconstructed_flat, original_flat))).item()
                        
                        ############# Computing losses (unitless) in Normalise space : shapes in [0,1] (Same as training) #######################
                        # # # Compute Chamfer Distance

                        # dist, _ = chamfer_dis(decoded.float(), inputs.float())
                        # # loss_value = dist.item()
                        # chamfer_value = dist.item()
                        # mse_value = criterion(decoded.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1)).item()
                        # rmse_value = (torch.sqrt(criterion(decoded.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1)))).item()
                        
                        ##############################################################################################################
                        
                        # print(f"{subject_id} - Average Chamfer Distance across the frames: {loss_value:.6f}")
                        # losses.append({"eid_18545": subject_id, "chamfer_distance": loss_value})
                        print(f"{subject_id} - Chamfer: {chamfer_value:.6f}, MSE: {mse_value:.6f}, RMSE: {rmse_value:.6f}")
                        losses.append({
                            "eid_18545": subject_id,
                            "chamfer_distance": chamfer_value,
                            "mse_loss": mse_value,
                            "rms_distance": rmse_value
                        })

                pbar.update(1)

    # Report average Chamfer Distance
    if rec and losses:
        
        # avg_chamfer = sum([d["chamfer_distance"] for d in losses]) / len(losses)
        # print(f"\n Average Chamfer Distance across test set: {avg_chamfer:.6f}")

        # avg_mse = sum([d["mse_loss"] for d in losses]) / len(losses)
        # print(f"\n Average MSE Distance across test set: {avg_mse:.6f}")
        
        # avg_rmse = sum([d["rms_distance"] for d in losses]) / len(losses)
        # print(f"\n Average RMSE Distance across test set: {avg_rmse:.6f}")
        
        
        # Extract values into numpy arrays
        chamfers = np.array([d["chamfer_distance"] for d in losses])
        mses = np.array([d["mse_loss"] for d in losses])
        rmses = np.array([d["rms_distance"] for d in losses])

        # Compute mean and standard deviation
        avg_chamfer, std_chamfer = chamfers.mean(), chamfers.std()
        avg_mse, std_mse = mses.mean(), mses.std()
        avg_rmse, std_rmse = rmses.mean(), rmses.std()

        # Print nicely
        print("\n=== Reconstruction Error (Real Space) ===")
        print(f"Chamfer Distance: {avg_chamfer:.4f} ± {std_chamfer:.4f}")
        print(f"MSE Distance:     {avg_mse:.4f} ± {std_mse:.4f}")
        print(f"RMSE Distance:    {avg_rmse:.4f} ± {std_rmse:.4f}")
                

        # Save Chamfer distances to CSV
        chamfer_df = pd.DataFrame(losses)

        # --- Append mean & std rows ---
        summary_rows = pd.DataFrame([
            {
                "eid_18545": "MEAN",
                "chamfer_distance": avg_chamfer,
                "mse_loss": avg_mse,
                "rms_distance": avg_rmse
            },
            {
                "eid_18545": "STD",
                "chamfer_distance": std_chamfer,
                "mse_loss": std_mse,
                "rms_distance": std_rmse
            }
        ])

        # Append and save
        chamfer_df = pd.concat([chamfer_df, summary_rows], ignore_index=True)

        # --- Save CSV ---
        chamfer_csv_path = os.path.join(output_dir, f"losses_test_{len(test_loader)}.csv")
        chamfer_df.to_csv(chamfer_csv_path, index=False)
        print(f"Distances and summary saved to {chamfer_csv_path}")

        chamfer_csv_path = os.path.join(output_dir, f"losses_test_{len(test_loader)}.csv")
        chamfer_df.to_csv(chamfer_csv_path, index=False)
        print(f"Distances saved to {chamfer_csv_path}")

    # Save embeddings to CSV
    embeddings = torch.vstack(embeddings_list).numpy()
    id_df = pd.DataFrame(subject_ids_list, columns=["eid_18545"])
    embeddings_df = pd.DataFrame(embeddings, columns=[f"c_{i+1}" for i in range(embeddings.shape[1])])
    combined_df = pd.concat([id_df, embeddings_df], axis=1)

    embeddings_file = os.path.join(output_dir, output_name)
    combined_df.to_csv(embeddings_file, index=False)
    print(f"Embeddings saved to {embeddings_file}")

    

    
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_latent_variable_impact_with_quantiles_latentvar(
    embeddings_path, output_dir, quantile_ranges, subject_ids_all, pos_all, 
    original_min, original_max, device
):
    """
    Analyze the impact of latent variables on LV morphology using quantiles, visualize results, 
    and save latent dimension variations.

    Args:
        embeddings_path: Path to the precomputed embeddings file.
        output_dir: Directory to save results.
        quantile_ranges: List of quantile ranges (e.g., [[0, 0.02], [0.095, 0.105], ...]).
        subject_ids_all: Subject IDs (aligned with embeddings).
        pos_all: Mesh positions for all subjects.
        original_min: Min value for reverse preprocessing (torch tensor).
        original_max: Max value for reverse preprocessing (torch tensor).
        device: Torch device (e.g., 'cuda' or 'cpu').
    """
    # Load embeddings
    embeddings_df = pd.read_csv(embeddings_path)  # Assuming CSV format
    embeddings = embeddings_df.iloc[:, 1:].values  # Exclude subject IDs
    latent_dim = embeddings.shape[1]  # Number of latent dimensions

    # Convert original_min and original_max to NumPy
    original_min = original_min.cpu().numpy()
    original_max = original_max.cpu().numpy()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Measure variation for each latent dimension
    variations = []
    for dim in range(latent_dim):
        latent_values = embeddings[:, dim]
        variation = np.std(latent_values)  # Standard deviation as variation measure
        variations.append((dim, variation))

    # Convert to DataFrame
    variations_df = pd.DataFrame(variations, columns=["latent_dim", "variation"])

    # Save variations to CSV
    variation_csv_path = os.path.join(output_dir, "latent_variations.csv")
    variations_df.to_csv(variation_csv_path, index=False)
    print(f"Saved latent variations to {variation_csv_path}")

    # Optional: filter dimensions based on a variation threshold
    variation_threshold = 1e-2  # you can adjust this
    valid_dims = variations_df[variations_df['variation'] > variation_threshold]['latent_dim'].tolist()

    print(f"Valid latent dimensions with significant variation: {valid_dims}")

    # Save valid_dims to a CSV file
    valid_dims_df = pd.DataFrame(valid_dims, columns=["valid_latent_dim"])
    valid_dims_csv_path = os.path.join(output_dir, "valid_latent_dimensions.csv")
    valid_dims_df.to_csv(valid_dims_csv_path, index=False)

    print(f"Saved valid latent dimensions to {valid_dims_csv_path}")
    
    
    # Analyze only valid dimensions
    for dim in tqdm(valid_dims, desc="Processing Latent Dimensions for Quantiles"):
        dim_output_dir = os.path.join(output_dir, f"latent_dim_{dim+1}")
        os.makedirs(dim_output_dir, exist_ok=True)

        # Extract latent values for the current dimension
        latent_values = embeddings[:, dim]

        # Compute quantile thresholds
        quantile_thresholds = [np.quantile(latent_values, q) for q in np.array(quantile_ranges).flatten()]

        # Plot the latent value distribution and quantile boundaries
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(latent_values, bins=50, alpha=0.7)
        for q_val in quantile_thresholds:
            ax.set_xlim([-2, 2])  # <--- ADD THIS

            ax.axvline(q_val, color='red', linestyle='--')
        ax.set_title(f"Latent Dimension {dim} Value Distribution")
        plt.savefig(os.path.join(dim_output_dir, f"latent_dim_{dim}_distribution.png"))
        plt.close(fig)

        # Process each quantile range
        for i in range(0, len(quantile_thresholds), 2):
            lower_bound, upper_bound = quantile_thresholds[i], quantile_thresholds[i + 1]

            # Select subjects in the current quantile range
            subject_indices = np.where((latent_values >= lower_bound) & (latent_values < upper_bound))[0]
            selected_meshes = pos_all[subject_indices]

            if len(selected_meshes) == 0:
                print(f"No subjects found in latent dimension {dim} range {lower_bound}-{upper_bound}.")
                continue

            # Unscale and rescale meshes
            unscaled_meshes = (selected_meshes - original_min) / (original_max - original_min)
            average_mesh = unscaled_meshes.mean(axis=0)
            rescaled_mesh = average_mesh * (original_max - original_min) + original_min

            # Save averaged shape
            quantile_folder = os.path.join(dim_output_dir, f"quantile_{i//2 + 1}")
            os.makedirs(quantile_folder, exist_ok=True)
            save_as_vtk_emb(torch.tensor(rescaled_mesh), f"quantile_{i//2 + 1}", quantile_folder, args)  # Provide args if needed

            # # Save plot of the mesh
            # fig = plt.figure(figsize=(10, 7))
            # ax = fig.add_subplot(111, projection="3d")
            # ax.scatter(rescaled_mesh[0, :, 0], rescaled_mesh[0, :, 1], rescaled_mesh[0, :, 2], s=2)
            # ax.set_title(f"Latent Dim {dim} Quantile {i//2 + 1} - Frame 0")
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # # set_axes_equal(ax)  # <-- ADD THIS LINE
            # ax.set_box_aspect([1,1,1])  # Equal aspect

            # # Set better viewing angle
            # ax.view_init(elev=30, azim=-60)
            # # ax.view_init(elev=10, azim=270)
            # plt.savefig(os.path.join(quantile_folder, f"quantile_{i//2 + 1}_frame_0.png"))
            # plt.close(fig)

            print(f"Saved results for latent dimension {dim}, quantile range {lower_bound}-{upper_bound}.")

    print("Quantile-based latent variable analysis completed.")


# def test_model_and_save_embeddings(model, model_path, test_loader, output_name, output_dir,args, rec):
#     """
#     Load a trained model, run it on test data, and save embeddings to a CSV file incrementally.

#     Args:
#         model_path: Path to the trained model checkpoint.
#         model: The model architecture.
#         test_loader: The test data loader.
#         output_file: Path to save the CSV file with embeddings.
#     """

#     import os  # For checking file existence

#     # Load the trained model
#     print("Loading the trained model...")
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#     print("Model loaded successfully!")

#     # Process test data and save embeddings incrementally
#     print("Processing test data and saving embeddings...")

#     embeddings_list = []  # Collect embeddings here
#     subject_ids_list = []  # Collect subject IDs
#     with torch.no_grad():
#         with tqdm(total=len(test_loader), desc="Processing Batches", unit="batch") as pbar:
#             for inputs, subject_ids, original_mins, original_maxs in test_loader:
#                 inputs = inputs.to(device)
#                 # encoded, decoded = model(inputs)
#                 mu, log_var, z, decoded = model(inputs)
                    
                    
#                 # Save embeddings and subject IDs
#                 # embeddings_list.append(encoded.detach().cpu())
#                 embeddings_list.append(mu.detach().cpu())
                    
#                 subject_ids_list.extend(subject_ids)  # Extend to handle batch size > 1
                
#                 if rec == True :
#                     print('Reconstruction.....')
#                     for i, subject_id in enumerate(subject_ids):
#                         original_shape = inputs[i].shape  # Shape of the original data   original: torch.Size([3, 50, 37, 38])
#                             # print("original:", inputs[i].shape )
#                         reconstructed = reverse_preprocessing(
#                                 decoded[i],
#                                 n_nodes,
#                                 original_shape,
#                                 original_mins[i],
#                                 original_maxs[i],
#                                 normalize=True
#                         )  ## reconstructed torch.Size([50, 1406, 3])
#                             # print("reconstructed", reconstructed.shape)
                            
#                         original = inputs[i].permute(1, 2, 3, 0).view(n_frames, -1, 3).cpu()  # Restore [T, Nodes, C] ## reconstructed torch.Size([50, 1406, 3])
#                         original = original[:, :n_nodes, :]        # Remove padding (if any)
#                         original = original * (original_maxs[i] - original_mins[i]) + original_mins[i]
#                         # print("original", original.shape)
                            
#                         # original = inputs[i].cpu().numpy()
#                         save_as_vtk(reconstructed.cpu() , original, subject_id, output_dir, args)
                        
#                 pbar.update(1)  # Update the progress bar for each batch


#     # Combine embeddings and save to CSV
#     embeddings = torch.vstack(embeddings_list).numpy()  # Stack collected tensors
#     id_df = pd.DataFrame(subject_ids_list, columns=["eid_18545"])
#     embeddings_df = pd.DataFrame(embeddings, columns=[f"c_{i+1}" for i in range(embeddings.shape[1])])

#     # Combine IDs with embeddings
#     combined_df = pd.concat([id_df, embeddings_df], axis=1)

#     # Save to CSV
    
#     embeddings_file = os.path.join(output_dir, output_name)
#     combined_df.to_csv(embeddings_file, index=False)
#     print(f"Embeddings saved to {embeddings_file}")    
    
#     # with torch.no_grad():
#     #     with tqdm(total=len(test_loader), desc="Processing Batches", unit="batch") as pbar:
#     #         for inputs, ids, _, _ in test_loader:
#     #             inputs = inputs.to(device)
#     #             # encoded, _ = model(inputs)  # Extract embeddings
#     #             mu, log_var, z, decoded = model(inputs)

#     #             # Convert embeddings and IDs to DataFrame
#     #             # embeddings = encoded.cpu().numpy()
#     #             if  MEAN == 'mu':
#     #                 embeddings = mu.cpu().numpy()
#     #             elif MEAN == 'z':
#     #                 print('z= mu+e*sigma')
#     #                 embeddings = z.cpu().numpy()
                
#     #             embeddings_df = pd.DataFrame(embeddings, columns=[f"c_{i+1}" for i in range(embeddings.shape[1])])
#     #             subject_ids_df = pd.DataFrame(ids, columns=["eid_18545"])
#     #             combined_df = pd.concat([subject_ids_df, embeddings_df], axis=1)

#     #             # Append to CSV file
#     #             if not os.path.exists(output_file):  # If file doesn't exist, write header
#     #                 combined_df.to_csv(output_file, mode='w', index=False)
#     #             else:  # Otherwise, append without header
#     #                 combined_df.to_csv(output_file, mode='a', index=False, header=False)

#     #             pbar.update(1)  # Update the progress bar for each batch

#     # print(f"Embeddings saved incrementally to {output_file}!")



# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train and test FactorVAE.")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device (e.g., 'cuda:1').")    
    parser.add_argument("--exp_name", type=str, default="VAE_r3d_18", help="Name of the method")## SPTVAE
    parser.add_argument("--encoded_dim", type=int, default=32, help="Latent space dimension.") ##default=128   16
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--lr", type=int, default=1e-4, help="Learning Rate.") ##1e-4
    parser.add_argument("--alpha", type=float, default=0.0001, help="Alpha for total smoothing laplacian loss.") ## default=0.01
    parser.add_argument("--gamma", type=float, default=0.0001, help="Gamma for normal loss.") ## default=0.01
    parser.add_argument("--delta", type=float, default=0.0001, help="temporal_smoothness")  ## default=1.0   0.001
    parser.add_argument("--beta", type=float, default=0.001, help="temporal_smoothness")  ## default=1.0   0.001
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="L2 regularization weight.")

    # parser.add_argument("--input_dir", type=str, default="/cardiac/sk/Space   MAP_deepmesh", help="Input directory.")  
    parser.add_argument("--input_dir", type=str, default="/ukb/sk/packdata_deepmesh/", help="Input directory.") 
    # parser.add_argument("--output_base_dir", type=str, default="/skalaie/motion_code/DR_motion/SPTVAE/results_VAE_optuna_10k/optuna_trial_25/", help="Output directory.")
    parser.add_argument("--output_base_dir", type=str, default="/skalaie/motion_code/DR_motion/SPTVAE/results_VAE_optuna_10k/VAE_32/", help="Output directory.")

    parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
    parser.add_argument("--best_trial_dir", type=str, default="/skalaie/motion_code/DR_motion/SPTVAE/results_VAE_optuna_10k/VAE_32/", help="Optuna trials directory.")

    # parser.add_argument("--atlas", type=str, default="/ukb/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec/avg/LVmyo_mesh_avg_decim_0.95_0.vtk", help="Atlas dir.")    
    parser.add_argument("--atlas", type=str, default="/ukb/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec_quadsim/avg/LVmyo_mesh_avg_decim_1200_0_quadsim.vtk", help="Atlas dir.")
    # parser.add_argument("--input_dir", type=str, default="S:/sk/SpaceMAP_deepmesh", help="Input directory.")
    # parser.add_argument("--output_base_dir", type=str, default="P:/motion_code/DR_motion/", help="Output directory.")
    # parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
    # parser.add_argument("--atlas", type=str, default="S:/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec/avg/LVmyo_mesh_avg_decim_0.95_0.vtk", help="Atlas dir.")
    
    return parser.parse_args()




def _init_():
    if not os.path.exists(args.output_base_dir):
        os.makedirs(args.output_base_dir)
            
    checkpoints_dir = os.path.join(args.output_base_dir, 'checkpoints/')    
    if not os.path.exists(args.output_base_dir +'checkpoints/'):
        os.makedirs(args.output_base_dir +'checkpoints/')


# python main_finetune_motion_factorVAE_new.py --latent_dim 64 --epochs 20 --gamma 5.0 

if __name__ == "__main__":
    
    args = parse_args()
    _init_()
    
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_base_dir, exist_ok=True)
    

    # print('Beta:',args.beta)
    # print('Alpha:',args.alpha)
    # print('LR:',args.lr)
    # ############  Test dummy data  ######################
    # N, T, Nodes, C = 100, 50, 1200, 3  # Example dimensions
    # pos = np.random.rand(N, T, Nodes, C)
    # n_frames = T
    # n_nodes = Nodes
    # # H = int(math.sqrt(Nodes))
    # H= 30
    # W = math.ceil(Nodes / H)
    # subject_ids = list(range(N))
    # train_size = int(0.9 * len(pos))
    # train_pos, val_pos = pos[:train_size], pos[train_size:]
    # train_ids, val_ids = subject_ids[:train_size], subject_ids[train_size:]
    # test_pos = pos
    # test_ids = subject_ids 

    ########## DeepMesh #######################################################################
    # NUM_SUBJECTS = 37039 ## with outliers
    # NUM_SUBJECTS = 36856
    NUM_SUBJECTS = 41257 ##41261
    print('NUM_SUBJECTS',NUM_SUBJECTS)
    # INPUT_DIR = "/cardiac/sk/SpaceMAP_deepmesh"
    # pickle_file_path = f'{INPUT_DIR}/id_pos_NS{NUM_SUBJECTS}_parallel_filtered_outliers.pkl'
    
    # pickle_file_path = f'{args.input_dir}/id_pos_NS{NUM_SUBJECTS}_parallel_filtered_outliers.pkl' ## 25 frames
    # pickle_file_path = f'{args.input_dir}/id_pos_NS{NUM_SUBJECTS}_FRstep_1_parallel_filtered_outliers.pkl' ## 50 frames
    
    # pickle_file_path = f'{args.input_dir}/id_pos_NS{NUM_SUBJECTS}_FRstep_1_parallel_filtered_outliers_quadsim.pkl' ## 50 frames
    pickle_file_path = f'{args.input_dir}/id_pos_NS{NUM_SUBJECTS}_FRstep_1_parallel_filtered_outliers_quadsim_batch2.pkl' ## 50 frames
    print(pickle_file_path)
    
    
    
    # id_pos_NS41261_FRstep_1_parallel_filtered_outliers_quadsim_batch2_resumable
    ########## SAX ###########################################################################
    # INPUT_DIR = "/cardiac/sk/SpaceMAP"
    # NUM_SUBJECTS = 10000
    # pickle_file_path = f'{INPUT_DIR}/id_pos_NS{NUM_SUBJECTS}_parallel.pkl'
    ##########################################################################################

    # # ######################################################################################
    # # ## Load new dataset without outliers 
    # # ######################################################################################
    # # with open(f'{INPUT_DIR}/id_pos_NS{NUM_SUBJECTS}_parallel_filtered_outliers.pkl', "rb") as f:
    # #     subject_ids_all, pos_all = pickle.load(f)
    ##################### Final ###########################################################################################    
    # # # Real data  #####
    # with open(pickle_file_path, "rb") as f:
    #     subject_ids_all, pos_all = pickle.load(f)

    ###########################################################################################################

    ###########################################################################################     
    #### All 80 k together :
    NUM_SUBJECTS_b1 = 36856
    NUM_SUBJECTS_b2 = 41257
    pickle_file_path_b1 = f'{args.input_dir}/id_pos_NS{NUM_SUBJECTS_b1}_FRstep_1_parallel_filtered_outliers_quadsim.pkl' ## 50 frames
    pickle_file_path_b2 = f'{args.input_dir}/id_pos_NS{NUM_SUBJECTS_b2}_FRstep_1_parallel_filtered_outliers_quadsim_batch2.pkl' ## 50 frames
    print(pickle_file_path_b1)
    print(pickle_file_path_b2)

    # # # # Real data  80 k #####
    with open(pickle_file_path_b1, "rb") as f:
        subject_ids_all_b1, pos_all_b1 = pickle.load(f)
    with open(pickle_file_path_b2, "rb") as f:
        subject_ids_all_b2, pos_all_b2 = pickle.load(f)
    
        
    # Merge 80k correctly
    subject_ids_all = subject_ids_all_b1 + subject_ids_all_b2                # list concat
    pos_all = np.concatenate([pos_all_b1, pos_all_b2], axis=0)               # array concat

    print(type(subject_ids_all), len(subject_ids_all))  # list
    print(type(pos_all), pos_all.shape)                 # numpy.ndarray

    # # pos = pos_all[:2000]
    # # subject_ids = subject_ids_all[:2000]
    
    
    # # Set random seed for reproducibility (optional)
    # np.random.seed(42)

    # # Choose the number of random samples
    # n_samples = 2000

    # # Get total available data size
    # total_samples = len(pos_all)

    # # Randomly select indices
    # random_indices = np.random.choice(total_samples, size=n_samples, replace=False)

    # # Convert to arrays and apply indexing
    # pos_all = np.array(pos_all)  # shape: [total_samples, T, nodes, 3]
    # subject_ids_all = np.array(subject_ids_all)

    # # Apply random sampling
    # pos = pos_all[random_indices]              # shape: [2000, T, nodes, 3]
    # subject_ids = subject_ids_all[random_indices]  # shape: [2000]
        
    # ##### 2000 data
    # # data = torch.load("subjects_2000_quadsim.pt")
    # # pos_all = data['pos']
    # # subject_ids_all = data['subject_ids']
    # # pos = pos_all
    # # subject_ids = subject_ids_all
    

    ############################# Test subjects ######################################

    # data = torch.load(f"{args.best_trial_dir}/test_data.pt")
    
    # pos_all = data['pos']
    # subject_ids_all = data['subject_ids']
    # pos_all = pos_all[0:20]
    # subject_ids_all = subject_ids_all[0:20]
    ##################################################################################
    
    N, n_frames, n_nodes, C = pos_all.shape   # pos.shape : [N,T,nodes,3]
    # H = int(math.sqrt(n_nodes))
    # W = math.ceil(n_nodes / H)
    H = 30
    W = math.ceil(n_nodes / H)
    print( "Number of Nodes", n_nodes)
    print( "Number of frames", n_frames)
    
    ##################################################################################################
    ### Split data
    # train_size = int(0.95 * len(pos))
    # train_pos, val_pos = pos[:train_size], pos[train_size:]
    # train_ids, val_ids = subject_ids[:train_size], subject_ids[train_size:]

    ### Whole population 
    test_pos = pos_all
    test_ids = subject_ids_all 

    # test_pos = val_pos
    # test_ids = val_ids


    # # # Datasets and loaders
    # train_dataset = MotionDataset(train_pos, train_ids)
    # val_dataset = MotionDataset(val_pos, val_ids)
    test_dataset = MotionDataset(test_pos, test_ids)
    
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader =DataLoader(test_dataset, batch_size=1, shuffle=False)

    # n_frames = 25 
    # n_frames = 50 
    # H = 34
    # W = 35
    # n_nodes = 1187
    ################################################################################################
    # model = Cardio4DVAE(encoded_dim=args.encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    # # optimizer_vae = torch.optim.Adam(model.parameters(), args.lr)
    # optimizer_vae = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args.lr,
    #     betas=(0.5, 0.99),
    #     weight_decay=args.weight_decay
    # )
    
    
    ###############################################################################################
    ## FineTuning and training start from thr optuna
    ###############################################################################################
    
    # best_trial_params_file = f"{args.best_trial_dir}/best_trial_params.json"
    # with open(best_trial_params_file, 'r') as f:
    #     best_params = json.load(f)
        
    
    
    # best_trial_number_file = f"{args.best_trial_dir}/best_trial_number.json"
    # with open(best_trial_number_file, 'r') as f:
    #     best_trial_data = json.load(f)
    # best_trial = best_trial_data["trial_number"]  # Extract the integer



    # best_trial_params_file = f"{args.best_trial_dir}/optuna_trial_1/trial_1.json"
    # with open(best_trial_params_file, 'r') as f:
    #     best_params = json.load(f)
        
    # best_trial = 1

    # # Apply the best trial hyperparameters to args
    # for key, value in best_params.items():
    #     if hasattr(args, key):
    #         setattr(args, key, value)
    #     else:
    #         print(f"Warning: 'args' has no attribute named '{key}' — skipping.")
    
    
    # print('Best Trial:', best_trial)
    # print('Alpha:',args.alpha)
    # print('Gamma:',args.gamma)    
    # print('Beta:',args.beta)
    # print('Delta:',args.delta)
    # print('latent_dim:',args.encoded_dim)
    # print('LR:',args.lr)
    
    
    # args.output_base_dir = os.path.join(args.best_trial_dir, f"optuna_trial_{best_trial}")

    
    # Initialize model
    model = Cardio4DVAE(encoded_dim=args.encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    


    ################################################################################################
    ## load the Optuna checkpoint and resume training
    ################################################################################################
    
    # modelName_best = "optuna_trial_12_epoch_30"
    modelName_best = f"VAE_r3d_18_epoch_30"

    model_path = os.path.join(args.best_trial_dir+ '/checkpoints', modelName_best)
    
    
    # modelName_best = f"optuna_trial_{best_trial}_epoch_30"

    # model_path = os.path.join(args.best_trial_dir, f'optuna_trial_{best_trial}/checkpoints', modelName_best)
    
    
    
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint["model_state_dict"])
    ## optimizer_ae.load_state_dict(checkpoint["optimizer_state_dict"]) ## Optimizer Momentum State Can Hurt Fine-Tuning
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)


    ###############################################################################################
    ## Train
    ###############################################################################################
    # modelName = "r3d_18_betavae_resEncfcDec"
    # modelName = "r3d_18_bvae_L2smooth"

    # modelName = "r3d_18_bvae_L2smooth_epoch_1"
    # modelName = "r3d_18_betavae_resEncConvDec"
    # modelName = "optuna_trial_43_epoch_30"
    # modelName = args.exp_name     ##"r3d_18_betavae_resEncfcDec_epoch_20"


    def check_z_variability(model, val_loader, num_samples=10):
        """Print z values for different inputs to check if they vary."""
        model.eval()
        
        with torch.no_grad():
            for i, (inputs, _, _, _) in enumerate(val_loader):
                if i >= num_samples:
                    break
                inputs = inputs.to(device)
                mu, log_var, z, _ = model(inputs)
                print(f"Sample {i}: Mean of `z`: {z.mean().item():.6f}, Std: {z.std().item():.6f}")

    # Call function to check if `z` is varying
    # check_z_variability(model, val_loader)


    def test_decoder_variability(model, val_loader):
        """Manually change `z` to see if decoder outputs change."""
        model.eval()
        
        with torch.no_grad():
            for i, (inputs, _, _, _) in enumerate(val_loader):
                if i >= 1:
                    break
                inputs = inputs.to(device)
                mu, log_var, z, _ = model(inputs)

                # Test different `z` values
                z_random = torch.randn_like(z)  # Completely random `z`
                z_scaled = z * 5  # Scale `z` to see effect

                decoded_original = model.decoder(z)  # Original output
                decoded_random = model.decoder(z_random)  # With random `z`
                decoded_scaled = model.decoder(z_scaled)  # With scaled `z`

                # Compare reconstructions
                print("Checking decoder behavior:")
                print(f"Original z -> Output Mean: {decoded_original.mean().item():.6f}")
                print(f"Random z -> Output Mean: {decoded_random.mean().item():.6f}")
                print(f"Scaled z -> Output Mean: {decoded_scaled.mean().item():.6f}")



    ###############################################################################
    ## Latent Traversal
    ###############################################################################
    
    # # Run the test
    # # test_decoder_variability(model, val_loader)
    # modelName = "r3d_18_betavae_resEncConvDec_epoch_30"
    # model_path = os.path.join(out_dir, modelName) 
    # # output_file = os.path.join(out_dir, "validation_embeddings_epoch_30.csv")
    # # out_dir_trav = out_dir +"/latent_traversal_shapes"
    # # os.makedirs(out_dir_trav, exist_ok=True)
    # latent_traversal(
    #     model=model,
    #     model_path = model_path,
    #     embeddings_path= f"{out_dir}/validation_embeddings_epoch_30.csv",
    #     output_dir = out_dir, ##out_dir_trav,
    #     original_shape=(3, 50, 34, 35),
    #     original_min=torch.tensor([-85.5211]),
    #     original_max=torch.tensor([55.7752]),
    #     n_nodes=1187,
    #     args=args,  # <- Pass your args object here
    #     steps=7  # [-3σ, -2σ, ..., +3σ]
    # )


    
    ###############################################################################################
    ## Test
    ###############################################################################################
    # # # modelName = "best_model_trial_12"
    # # modelName = "optuna_trial_27_epoch_30"

    
    # # model_path = os.path.join(args.output_base_dir +'checkpoints/', modelName) 

    
    # # # # output_name = f"LVmyo_motion_embeddings_mu_{args.encoded_dim}_NS_{NUM_SUBJECTS}.csv"
    # # ## output_name = f"LVmyo_motion_embeddings_{args.encoded_dim}_NS_{NUM_SUBJECTS}.csv"
    # # ## output_file = os.path.join(out_dir, output_name)
    
    
    # ### Test whole data
    # print('Test on whole UKB data')
    # # output_name = f"LVmyo_motion_embeddings_{args.encoded_dim}_AE_NS_{len(test_dataset)}.csv"
    # output_name = f"LVmyo_motion_embeddings_{args.encoded_dim}_VAE_NS_{len(test_dataset)}.csv"
    # # output_name = f"LVmyo_motion_embeddings_{args.encoded_dim}_VAE_NS_{len(test_dataset)}_batch2.csv"
    # # output_name = f"LVmyo_motion_embeddings_{args.encoded_dim}_VAE_NS_{len(test_dataset)}_batch1.csv"


    # test_model_and_save_embeddings(model, model_path, test_loader,output_name, args.output_base_dir,args, rec= False)

    
    # ### check the reconstruction
    # # print('Validation')
    # # output_name = f"LVmyo_motion_embeddings_{args.encoded_dim}_AE_NS_{len(val_dataset)}.csv"
    # # test_model_and_save_embeddings(model, model_path, val_loader, output_name, args.output_base_dir,args, rec =True)

      ###############################################################################################
    ## Latent representation
    ###############################################################################################
    # averaged_shapes_dir = f"{out_dir}/averaged_shapes_quantile_20"
    # latent_dim_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # Latent dimensions to visualize
    # quantile_ranges = [[0, 0.01], [0.095, 0.105], [0.495, 0.505], [0.895, 0.905], [0.99, 1]]  # Quantile ranges  upper - lower = 0.01
    
    # quantile_ranges = [[0, 0.05], [0.05, 0.25], [0.25, 0.75], [0.75, 0.95], [0.95, 1]]
    
    
    quantile_ranges = [
    [0.00, 0.02],     # around the 1st percentile
    [0.09, 0.11],     # around the 10th percentile
    [0.49, 0.51],     # around the 50th percentile
    [0.89, 0.91],     # around the 90th percentile
    [0.98, 1.00]      # around the 99th percentile
    ] ### upper - lower = 0.02
    

    output_dir = f"{args.output_base_dir}/averaged_shapes_quantile_width0.02"  # Directory to save visualization figures
    
    embeddings_path = f"{args.output_base_dir}/LVmyo_motion_embeddings_32_VAE_NS_78113.csv" 
    
    # Run Analysis
    # analyze_latent_variable_impact_with_quantiles(
    #     embeddings_path, test_loader, output_dir, quantile_ranges, 
    #     original_shape, n_nodes, device
    # )
    
    # analyze_latent_variable_impact_with_quantiles(
    # embeddings_path, output_dir, quantile_ranges, subject_ids_all, pos_all, 
    # original_min, original_max, device)
    
    
    original_min =  torch.tensor([-85.5211])
    original_max =  torch.tensor([55.7752])
    
    
    analyze_latent_variable_impact_with_quantiles_latentvar(
    embeddings_path, output_dir, quantile_ranges, subject_ids_all, pos_all, 
    original_min, original_max, device
)
    