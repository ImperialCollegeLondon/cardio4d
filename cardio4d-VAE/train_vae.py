import pyvista as pv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.models.video import r3d_18
import pickle
import pandas as pd
import os
import time

import numpy as np
import math
from tqdm import tqdm
import argparse
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance as chamfer_dis
from pytorch3d.loss import mesh_normal_consistency as normal_dis
from pytorch3d.loss import mesh_laplacian_smoothing as lap_dis

import torch.nn.functional as F
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice
import plotly.graph_objects as go

import plotly.io as pio
import json

import random
from utils import MotionDataset
from utils import compute_total_correlation, beta_ae_lossT, beta_tcvae_lossT, beta_vae_lossT, beta_vae_loss
from utils import save_checkpoint, compute_latent_correlation, adaptive_beta, save_loss_plot, plot_training_curves_ae,plot_training_curves_ae_all, plot_training_curves_all
from utils import reverse_preprocessing, save_as_vtk, save_as_vtk_emb, save_as_vtk_emb_smooth
from model import AE, Cardio4DVAE

# def train_AE_smooth(model, modelName, train_loader, val_loader, optimizer, output_dir, n_frames, n_nodes, args):
    
#     """
#     Train the autoencoder with validation and save validation reconstructions as VTK files.
#     """

    
#     # Store values for visualization
#     alpha_values = []
#     gamma_values = []
#     train_losses, val_losses = [], []
#     train_norm_losses,  train_recon_losses, train_lap_losses = [], [], []
#     val_norm_losses,  val_recon_losses, val_lap_losses = [], [], []
#     model.train()
    
    
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#     ##============================================================
#     # Start with low values for alpha and beta
#     alpha = args.alpha
#     beta = args.beta
#     gamma = args.gamma
    

    
#     ##============================================================
    
#     for epoch in range(args.epochs):
#         model.train()
#         # Training phase
#         total_train_loss = 0.0
#         total_train_recon_loss = 0.0
#         total_train_lap_loss = 0.0
#         total_train_norm_loss = 0.0
#         ## Gradually increase alpha and beta
#         ## alpha = min(alpha_max, (epoch / total_epochs) * alpha_max)  # Linear warm-up for alpha
#         ## beta = min(beta_max, epoch * beta_step)  # Gradual increase of beta
        
#         # alpha = min(1.0, epoch * 0.01)  # Increase gradually
#         # beta = min(3e-5, epoch * 0.001)  # Increase gradually
        
        
#         # alpha_start = 0.1  # Start from 0.1 instead of 0
#         # beta_start = 1e-6  # Start from a lower beta

#         # alpha = min(1.0, alpha_start * (1.1 ** epoch))  # Exponential increase
#         # beta = min(3e-5, beta_start * (1.05 ** epoch))  # Exponential increase
        
        
#         # Dynamic Beta
#         # beta = adaptive_beta(epoch, start_beta=2e-5, max_beta=args.beta, warmup_epochs=15)
#         # beta = adaptive_beta(epoch, start_beta=1e-6, max_beta=args.beta, warmup_epochs=15)

        
#         for inputs, _, _, _ in train_loader:
#             inputs = inputs.to(device)  ## input size: [batch_size, C, T, H, W]  torch.Size([8, 3, 50, 37, 38])
#             # print("input", inputs.shape)
#             # encoded, decoded = model(inputs)
#             encoded, decoded = model(inputs)
#             # print("decoded", decoded.shape)  ## decoded torch.Size([8, 3, 50, 37, 38])
#             # loss = criterion(decoded.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1))  # Flatten inputs
#             # loss, recon_loss, kl_loss = beta_vae_loss(decoded, inputs, mu, log_var, beta)
#             loss, recon_loss,  lap_loss, norm_loss = beta_ae_lossT(decoded, inputs, n_nodes, alpha, beta,  gamma, args.atlas, args) 
#             # loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss = beta_tcvae_lossT(decoded, inputs, mu, log_var, z, n_nodes, alpha, beta, args.gamma, args.delta, args.atlas)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()
#             total_train_recon_loss += recon_loss.item()
#             total_train_lap_loss += lap_loss.item()
#             total_train_norm_loss += norm_loss.item()

#         ## Step the scheduler
#         scheduler.step()
        
#         # Validation phase
#         model.eval()
#         total_val_loss = 0.0
#         total_val_recon_loss = 0.0
#         total_val_lap_loss = 0.0
#         total_val_norm_loss = 0.0
#         total_val_sparse_loss = 0.0

#         embeddings_list = []  # Collect embeddings here
#         subject_ids_list = []  # Collect subject IDs
#         with torch.no_grad():
#             for inputs, subject_ids, original_mins, original_maxs in val_loader:
#                 inputs = inputs.to(device)
#                 # encoded, decoded = model(inputs)
#                 encoded, decoded = model(inputs)
                
                
#                 ## Check if z is Actually Different for Different Inputs
#                 # print(f"Sample {subject_ids}: Mean of `mu`: {mu.mean().item():.6f}, Std: {mu.std().item():.6f}")
                
#                 # loss = criterion(decoded.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1)) 
#                 # loss, recon_loss, kl_loss = beta_vae_loss(decoded, inputs, mu, log_var, beta)
#                 loss, recon_loss,  lap_loss, norm_loss = beta_ae_lossT(decoded, inputs,  n_nodes, alpha, beta, gamma, args.atlas, args)
#                 # loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss = beta_tcvae_lossT(decoded, inputs, mu, log_var, z, n_nodes, alpha, beta, args.gamma, args.delta, args.atlas)
#                 total_val_loss += loss.item()
#                 total_val_recon_loss += recon_loss.item()
#                 total_val_lap_loss += lap_loss.item()
#                 total_val_norm_loss += norm_loss.item()
                
                
#         #         # Save embeddings and subject IDs
#         #         # embeddings_list.append(encoded.detach().cpu())
#         #         embeddings_list.append(encoded.detach().cpu())
                
#         #         subject_ids_list.extend(subject_ids)  # Extend to handle batch size > 1
#         #         for i, subject_id in enumerate(subject_ids):
#         #             original_shape = inputs[i].shape  # Shape of the original data   original: torch.Size([3, 50, 37, 38])
#         #             # print("original:", inputs[i].shape )
#         #             reconstructed = reverse_preprocessing(
#         #                 decoded[i],
#         #                 n_nodes,
#         #                 original_shape,
#         #                 original_mins[i],
#         #                 original_maxs[i],
#         #                 normalize=True
#         #             )  ## reconstructed torch.Size([50, 1406, 3])
#         #             # print("reconstructed", reconstructed.shape)
                    
#         #             original = inputs[i].permute(1, 2, 3, 0).view(n_frames, -1, 3).cpu()  # Restore [T, Nodes, C] ## reconstructed torch.Size([50, 1406, 3])
#         #             original = original[:, :n_nodes, :]        # Remove padding (if any)
#         #             original = original * (original_maxs[i] - original_mins[i]) + original_mins[i]
#         #             # print("original", original.shape)
                    
#         #             # original = inputs[i].cpu().numpy()
#         #             save_as_vtk(reconstructed.cpu() , original, subject_id, output_dir, args)

#         # # Combine embeddings and save to CSV
#         # embeddings = torch.vstack(embeddings_list).numpy()  # Stack collected tensors
#         # id_df = pd.DataFrame(subject_ids_list, columns=["eid_18545"])
#         # embeddings_df = pd.DataFrame(embeddings, columns=[f"c_{i+1}" for i in range(embeddings.shape[1])])

#         # # Combine IDs with embeddings
#         # combined_df = pd.concat([id_df, embeddings_df], axis=1)

#         # # Save to CSV
#         # embeddings_file = os.path.join(output_dir, f"validation_embeddings_epoch_{epoch+1}.csv")
#         # combined_df.to_csv(embeddings_file, index=False)
#         # print(f"Embeddings saved to {embeddings_file}")


#         # Print epoch losses
#         avg_train_loss = total_train_loss / len(train_loader)
#         avg_val_loss = total_val_loss / len(val_loader)
#         avg_train_recon_loss = total_train_recon_loss / len(train_loader)
#         avg_train_lap_loss = total_train_lap_loss / len(train_loader)
#         avg_val_recon_loss = total_val_recon_loss / len(val_loader)
#         avg_val_lap_loss = total_val_lap_loss / len(val_loader)
#         avg_train_norm_loss = total_train_norm_loss / len(train_loader)
#         avg_val_norm_loss = total_val_norm_loss / len(val_loader)

#         alpha_values.append(alpha)
#         gamma_values.append(gamma)
        
#         # Append losses to lists
#         train_losses.append(avg_train_loss)
#         val_losses.append(avg_val_loss)
#         train_recon_losses.append(avg_train_recon_loss)
#         train_lap_losses.append(avg_train_lap_loss)
#         train_norm_losses.append(avg_train_norm_loss)
#         val_recon_losses.append(avg_val_recon_loss)
#         val_lap_losses.append(avg_val_lap_loss)
#         val_norm_losses.append(avg_val_norm_loss)
        
          
#         print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
#         ### ===========================================================================
#         ### Save weight ===============================================================

#         save_checkpoint(output_dir, modelName, model, optimizer, train_losses, val_losses, args.epochs)
#         # Save plot every 10 epochs
#         if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
#             # save_loss_plot(train_losses, val_losses, output_dir, epoch + 1, name="Total_loss")    
#             # save_loss_plot(train_recon_losses, val_recon_losses, output_dir, epoch + 1, name="Rec_loss")   
#             # save_loss_plot(train_kl_losses, val_kl_losses, output_dir, epoch + 1, name="KL_loss")    
#             plot_training_curves_ae(alpha_values, gamma_values, train_losses, val_losses, train_recon_losses, val_recon_losses, train_lap_losses, val_lap_losses, output_dir)

#             save_checkpoint(output_dir, f"{modelName}_epoch_{epoch+1}", model, optimizer, train_losses, val_losses, args.epochs)

#     print('==============')
#     print('Well Done!')
#     print('==============')
#     return val_losses , val_recon_losses


def train_AE_smooth(model, modelName, train_loader, val_loader, optimizer, output_dir, n_frames, n_nodes, args):
    
    """
    Train the autoencoder with validation and save validation reconstructions as VTK files.
    """

    
    # Store values for visualization
    alpha_values = []
    gamma_values = []
    train_losses, val_losses = [], []
    train_norm_losses,  train_recon_losses, train_lap_losses,  train_t_losses = [], [], [], []
    val_norm_losses,  val_recon_losses, val_lap_losses,  val_t_losses = [], [], [], []
    model.train()
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    ##============================================================
    # Start with low values for alpha and beta
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    

    
    ##============================================================
    
    for epoch in range(args.epochs):
        model.train()
        # Training phase
        total_train_loss = 0.0
        total_train_recon_loss = 0.0
        total_train_lap_loss = 0.0
        total_train_norm_loss = 0.0
        total_train_t_loss = 0.0

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
            loss, recon_loss,  lap_loss, norm_loss, t_loss = beta_ae_lossT(decoded, inputs, n_nodes, alpha, beta,  gamma, args.atlas, args) 
            # loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss = beta_tcvae_lossT(decoded, inputs, mu, log_var, z, n_nodes, alpha, beta, args.gamma, args.delta, args.atlas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_recon_loss += recon_loss.item()
            total_train_lap_loss += lap_loss.item()
            total_train_norm_loss += norm_loss.item()
            total_train_t_loss += t_loss.item()


        ## Step the scheduler
        scheduler.step()
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_recon_loss = 0.0
        total_val_lap_loss = 0.0
        total_val_norm_loss = 0.0
        total_val_t_loss = 0.0

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
                loss, recon_loss,  lap_loss, norm_loss, t_loss = beta_ae_lossT(decoded, inputs,  n_nodes, alpha, beta, gamma, args.atlas, args)
                # loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss = beta_tcvae_lossT(decoded, inputs, mu, log_var, z, n_nodes, alpha, beta, args.gamma, args.delta, args.atlas)
                total_val_loss += loss.item()
                total_val_recon_loss += recon_loss.item()
                total_val_lap_loss += lap_loss.item()
                total_val_norm_loss += norm_loss.item()
                total_val_t_loss += t_loss.item()
                
        #         # Save embeddings and subject IDs
        #         # embeddings_list.append(encoded.detach().cpu())
        #         embeddings_list.append(encoded.detach().cpu())
                
        #         subject_ids_list.extend(subject_ids)  # Extend to handle batch size > 1
        #         for i, subject_id in enumerate(subject_ids):
        #             original_shape = inputs[i].shape  # Shape of the original data   original: torch.Size([3, 50, 37, 38])
        #             # print("original:", inputs[i].shape )
        #             reconstructed = reverse_preprocessing(
        #                 decoded[i],
        #                 n_nodes,
        #                 original_shape,
        #                 original_mins[i],
        #                 original_maxs[i],
        #                 normalize=True
        #             )  ## reconstructed torch.Size([50, 1406, 3])
        #             # print("reconstructed", reconstructed.shape)
                    
        #             original = inputs[i].permute(1, 2, 3, 0).view(n_frames, -1, 3).cpu()  # Restore [T, Nodes, C] ## reconstructed torch.Size([50, 1406, 3])
        #             original = original[:, :n_nodes, :]        # Remove padding (if any)
        #             original = original * (original_maxs[i] - original_mins[i]) + original_mins[i]
        #             # print("original", original.shape)
                    
        #             # original = inputs[i].cpu().numpy()
        #             save_as_vtk(reconstructed.cpu() , original, subject_id, output_dir, args)

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
        avg_train_recon_loss = total_train_recon_loss / len(train_loader)
        avg_train_lap_loss = total_train_lap_loss / len(train_loader)
        avg_train_norm_loss = total_train_norm_loss / len(train_loader)
        avg_train_t_loss = total_train_t_loss / len(train_loader)
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_recon_loss = total_val_recon_loss / len(val_loader)
        avg_val_lap_loss = total_val_lap_loss / len(val_loader)
        avg_val_norm_loss = total_val_norm_loss / len(val_loader)
        avg_val_t_loss = total_val_t_loss / len(val_loader)
        
        alpha_values.append(alpha)
        gamma_values.append(gamma)
        
        # Append losses to lists
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_train_recon_loss)
        train_lap_losses.append(avg_train_lap_loss)
        train_norm_losses.append(avg_train_norm_loss)
        train_t_losses.append(avg_train_t_loss)
        
        val_losses.append(avg_val_loss)
        val_recon_losses.append(avg_val_recon_loss)
        val_lap_losses.append(avg_val_lap_loss)
        val_norm_losses.append(avg_val_norm_loss)
        val_t_losses.append(avg_val_t_loss)
          
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        ### ===========================================================================
        ### Save weight ===============================================================

        save_checkpoint(output_dir, modelName, model, optimizer, train_losses, val_losses, args.epochs)
        # Save plot every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            # save_loss_plot(train_losses, val_losses, output_dir, epoch + 1, name="Total_loss")    
            # save_loss_plot(train_recon_losses, val_recon_losses, output_dir, epoch + 1, name="Rec_loss")   
            # save_loss_plot(train_kl_losses, val_kl_losses, output_dir, epoch + 1, name="KL_loss")    
            # plot_training_curves_ae(alpha_values, gamma_values, train_losses, val_losses, train_recon_losses, val_recon_losses, train_lap_losses, val_lap_losses, train_norm_losses, val_norm_losses,train_t_losses, val_t_losses, output_dir)
            plot_training_curves_ae_all( train_losses, val_losses, train_recon_losses, val_recon_losses, train_lap_losses, val_lap_losses, train_norm_losses, val_norm_losses,train_t_losses, val_t_losses, output_dir)

            save_checkpoint(output_dir, f"{modelName}_epoch_{epoch+1}", model, optimizer, train_losses, val_losses, args.epochs)

    print('==============')
    print('Well Done!')
    print('==============')
    return val_losses , val_recon_losses



   
def train_betavae_smooth(model, modelName, train_loader, val_loader, optimizer, output_dir, n_frames, n_nodes, args):
    
    """
    Train the autoencoder with validation and save validation reconstructions as VTK files.
    """

    
    # Store values for visualization
    alpha_values = []
    beta_values = []
    train_losses, val_losses = [], []
    train_kl_losses, train_recon_losses, train_lap_losses, train_norm_losses, train_t_losses, = [], [], [], [], []
    val_kl_losses, val_recon_losses, val_lap_losses, val_norm_losses, val_t_losses = [], [], [],[],[]
    model.train()
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    ##============================================================
    # Start with low values for alpha and beta
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    delta = args.delta
    

    
    ##============================================================
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        # Training phase
        total_train_loss = 0.0
        total_train_kl_loss = 0.0
        total_train_recon_loss = 0.0
        total_train_lap_loss = 0.0
        total_train_norm_loss = 0.0
        total_train_t_loss = 0.0
        
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
        # beta = adaptive_beta(epoch, start_beta=0, max_beta=args.beta, warmup_epochs=15)
        # print('Beta:', beta)

        # beta = adaptive_beta(
        #     epoch,
        #     start_beta=1e-6,
        #     max_beta=args.beta,           # Final target value (e.g. 1.0)
        #     warmup_epochs=15,             # First 15 epochs: no KL
        #     grow_epochs=10,               # Next 10 epochs: linearly increase beta
        #     mode="linear"                 # Or "exponential"
        # )
        # print(f"Epoch {epoch+1}, Beta: {beta:.6f}")
        
        for inputs, _, _, _ in train_loader:
            inputs = inputs.to(device)  ## input size: [batch_size, C, T, H, W]  torch.Size([8, 3, 50, 37, 38])
            # print("input", inputs.shape)
            # encoded, decoded = model(inputs)
            mu, log_var, z, decoded = model(inputs)
            # print("Value range:", decoded.min().item(), decoded.max().item())
            # print("decoded", decoded.shape)  ## decoded torch.Size([8, 3, 50, 37, 38])
            # loss = criterion(decoded.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1))  # Flatten inputs
            # loss, recon_loss, kl_loss = beta_vae_loss(decoded, inputs, mu, log_var, beta)
            
            # loss, recon_loss, kl_loss, lap_loss = beta_vae_lossT(decoded, inputs, mu, log_var, n_nodes, alpha, beta, gamma, args.atlas, args)
            loss, recon_loss, kl_loss, lap_loss, norm_loss, t_loss = beta_vae_lossT(decoded, inputs, mu, log_var, n_nodes, alpha, beta, gamma, delta, args.atlas, args)

            # loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss = beta_tcvae_lossT(decoded, inputs, mu, log_var, z, n_nodes, alpha, beta, args.gamma, args.delta, args.atlas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_recon_loss += recon_loss.item()
            total_train_kl_loss += kl_loss.item()
            total_train_lap_loss += lap_loss.item()
            total_train_norm_loss += norm_loss.item()
            total_train_t_loss += t_loss.item()
            
        ## Step the scheduler
        scheduler.step()
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_recon_loss = 0.0
        total_val_kl_loss = 0.0
        total_val_lap_loss = 0.0
        total_val_norm_loss = 0.0
        total_val_t_loss = 0.0        
        
        embeddings_list = []  # Collect embeddings here
        subject_ids_list = []  # Collect subject IDs
        with torch.no_grad():
            for inputs, subject_ids, original_mins, original_maxs in val_loader:
                inputs = inputs.to(device)
                # encoded, decoded = model(inputs)
                mu, log_var, z, decoded = model(inputs)
                
                
                ## Check if z is Actually Different for Different Inputs
                # print(f"Sample {subject_ids}: Mean of `mu`: {mu.mean().item():.6f}, Std: {mu.std().item():.6f}")
                
                # loss = criterion(decoded.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1)) 
                # loss, recon_loss, kl_loss = beta_vae_loss(decoded, inputs, mu, log_var, beta)
                
                # loss, recon_loss, kl_loss, lap_loss = beta_vae_lossT(decoded, inputs, mu, log_var, n_nodes, alpha, beta, gamma, args.atlas, args)
                loss, recon_loss, kl_loss, lap_loss, norm_loss, t_loss = beta_vae_lossT(decoded, inputs, mu, log_var, n_nodes, alpha, beta, gamma,delta, args.atlas, args)

                # loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss = beta_tcvae_lossT(decoded, inputs, mu, log_var, z, n_nodes, alpha, beta, args.gamma, args.delta, args.atlas)
                total_val_loss += loss.item()
                total_val_recon_loss += recon_loss.item()
                total_val_kl_loss += kl_loss.item()
                total_val_lap_loss += lap_loss.item()
                total_val_norm_loss += norm_loss.item()
                total_val_t_loss += t_loss.item()
                

                # Save embeddings and subject IDs
                # embeddings_list.append(encoded.detach().cpu())
                embeddings_list.append(z.detach().cpu())
                
            #     subject_ids_list.extend(subject_ids)  # Extend to handle batch size > 1
            #     for i, subject_id in enumerate(subject_ids):
            #         original_shape = inputs[i].shape  # Shape of the original data   original: torch.Size([3, 50, 37, 38])
            #         # print("original:", inputs[i].shape )
            #         reconstructed = reverse_preprocessing(
            #             decoded[i],
            #             n_nodes,
            #             original_shape,
            #             original_mins[i],
            #             original_maxs[i],
            #             normalize=True ## False
            #         )  ## reconstructed torch.Size([50, 1406, 3])
            #         # print("reconstructed", reconstructed.shape)
                    
            #         original = inputs[i].permute(1, 2, 3, 0).view(n_frames, -1, 3).cpu()  # Restore [T, Nodes, C] ## reconstructed torch.Size([50, 1406, 3])
            #         original = original[:, :n_nodes, :]        # Remove padding (if any)
            #         original = original * (original_maxs[i] - original_mins[i]) + original_mins[i]
            #         # print("original", original.shape)
                    
            #         # original = inputs[i].cpu().numpy()
            #         save_as_vtk(reconstructed.cpu() , original, subject_id, output_dir, args)
            # print("Value range, last valiadtion:", decoded.min().item(), decoded.max().item())
        # # Combine embeddings and save to CSV
        # embeddings = torch.vstack(embeddings_list).numpy()  # Stack collected tensors
        # id_df = pd.DataFrame(subject_ids_list, columns=["eid_18545"])
        # embeddings_df = pd.DataFrame(embeddings, columns=[f"c_{i+1}" for i in range(embeddings.shape[1])])

        # # Combine IDs with embeddings
        # combined_df = pd.concat([id_df, embeddings_df], axis=1)

        ### Save to CSV
        # embeddings_file = os.path.join(output_dir, f"validation_embeddings_epoch_{epoch+1}.csv")
        # combined_df.to_csv(embeddings_file, index=False)
        # print(f"Embeddings saved to {embeddings_file}")


        # Print epoch losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_recon_loss = total_train_recon_loss / len(train_loader)
        avg_train_kl_loss = total_train_kl_loss / len(train_loader)
        avg_train_lap_loss = total_train_lap_loss / len(train_loader)
        avg_train_norm_loss = total_train_norm_loss / len(train_loader)
        avg_train_t_loss = total_train_t_loss / len(train_loader)
        
        avg_val_loss = total_val_loss / len(val_loader)        
        avg_val_recon_loss = total_val_recon_loss / len(val_loader)
        avg_val_kl_loss = total_val_kl_loss / len(val_loader)
        avg_val_lap_loss = total_val_lap_loss / len(val_loader)
        avg_val_norm_loss = total_val_norm_loss / len(val_loader)
        avg_val_t_loss = total_val_t_loss / len(val_loader)
        
                        
        alpha_values.append(alpha)
        beta_values.append(beta)
        
        # Append losses to lists
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_train_recon_loss)
        train_kl_losses.append(avg_train_kl_loss)
        train_lap_losses.append(avg_train_lap_loss)
        train_norm_losses.append(avg_train_norm_loss)
        train_t_losses.append(avg_train_t_loss)
        
        val_losses.append(avg_val_loss)        
        val_recon_losses.append(avg_val_recon_loss)
        val_kl_losses.append(avg_val_kl_loss)
        val_lap_losses.append(avg_val_lap_loss)
        val_norm_losses.append(avg_val_norm_loss)
        val_t_losses.append(avg_val_t_loss)
        
        # print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        epoch_end = time.time()  
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} -- took {epoch_end - epoch_start:.2f} seconds")
        ### ===========================================================================
        ### Save weight ===============================================================
        # compute_latent_correlation(output_dir, f"latent_corr_epoch_{epoch+1}", embeddings_list)
        # save_checkpoint(output_dir, modelName, model, optimizer, train_losses, val_losses, args.epochs)
        save_checkpoint(output_dir, f"{modelName}_epoch_{epoch+1}", model, optimizer, train_losses, val_losses, args.epochs)

        # Save plot every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            # save_loss_plot(train_losses, val_losses, output_dir, epoch + 1, name="Total_loss")    
            # save_loss_plot(train_recon_losses, val_recon_losses, output_dir, epoch + 1, name="Rec_loss")   
            # save_loss_plot(train_kl_losses, val_kl_losses, output_dir, epoch + 1, name="KL_loss") 
               
            # plot_training_curves(alpha_values, beta_values, train_losses, val_losses, train_recon_losses, val_recon_losses, train_kl_losses, val_kl_losses, train_lap_losses, val_lap_losses, output_dir)
            plot_training_curves_all( beta_values, train_losses, val_losses, train_recon_losses, val_recon_losses, train_kl_losses, val_kl_losses, train_lap_losses,val_lap_losses,  train_norm_losses, val_norm_losses, train_t_losses, val_t_losses, output_dir)
            
            compute_latent_correlation(output_dir, f"latent_corr_epoch_{epoch+1}", embeddings_list)

            save_checkpoint(output_dir, f"{modelName}_epoch_{epoch+1}", model, optimizer, train_losses, val_losses, args.epochs)
    print('==============')
    print('Well Done!')
    print('==============')
    # return val_losses , val_recon_losses
    return val_losses , val_recon_losses, val_kl_losses, val_lap_losses, val_norm_losses, val_t_losses








import pandas as pd
from tqdm import tqdm

def calculate_embedding_importance(model, model_path, test_loader, criterion, output_file, device):
    """
    Calculate the importance of each embedding dimension based on reconstruction loss.

    Args:
        model: I3DAutoencoder model (architecture).
        model_path: Path to the trained model checkpoint.
        test_loader: DataLoader for test data.
        criterion: Loss function (e.g., MSELoss).
        output_file: Path to save the importance scores as a CSV file.
        device: Torch device (CPU or GPU).

    Returns:
        A DataFrame containing importance scores for each embedding dimension.
    """
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Store importance scores for all batches
    all_importance_scores = []

    # Loop over test data
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Processing Batches", unit="batch") as pbar:
            for inputs, ids, _, _ in test_loader:
                inputs = inputs.to(device)

                # Get embeddings and original reconstruction loss
                embeddings, decoded = model(inputs)
                original_loss = criterion(decoded.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1)) .item()

                # Store importance scores for this batch
                batch_importance_scores = []

                embedding_dim = embeddings.size(1)  # Number of embedding dimensions
                for dim in range(embedding_dim):
                    # Perturb or mask the current embedding dimension
                    modified_embeddings = embeddings.clone()
                    modified_embeddings[:, dim] = 0  # Mask the dimension

                    # Compute reconstruction loss with the modified embeddings
                    decoded_modified = model.decoder(modified_embeddings)  # Decode from modified embeddings
                    modified_loss = criterion(decoded_modified.view(inputs.size(0), -1), inputs.view(inputs.size(0), -1)).item()
                    # Compute the difference in loss (importance score)
                    batch_importance_scores.append(modified_loss - original_loss)

                # Append batch scores to the global list
                all_importance_scores.append(batch_importance_scores)
                pbar.update(1)

    # Convert to DataFrame
    importance_scores_df = pd.DataFrame(
        all_importance_scores,
        columns=[f"c_{i+1}" for i in range(embedding_dim)]
    )

    # Save to CSV
    importance_scores_df.to_csv(output_file, index=False)
    print(f"Importance scores saved to {output_file}")

    return all_importance_scores







'''

To analyze the effect of each embedding dimension at its mean ± 3 standard deviations, similar to PCA:

1. Calculate the mean and standard deviation of each embedding dimension across all subjects.
2. Perturb each dimension in the embedding space by adding/subtracting 3 times its standard deviation while keeping other dimensions fixed at their mean.
3. Reconstruct shapes using the perturbed embeddings.


Steps:
1. Calculate mean and standard deviation for each embedding dimension.
2. Generate perturbed embeddings for each dimension at mean ± 3 SD.
3. Use the perturbed embeddings for reconstruction.

'''


def vary_latent_components_and_generate_shapes(
    model, model_path, embeddings_path, output_dir, latent_dim_range, 
    original_min, original_max, original_shape, n_nodes, device
):
    """
    
    Shapes are back to subject space (denormalize)
    Vary each latent component one at a time and generate synthetic shapes.

    Args:
        model: Trained autoencoder model.
        model_path: Path to the trained model checkpoint.
        embeddings_path: Path to the file containing precomputed embeddings.
        output_dir: Directory to save generated synthetic shapes.
        latent_dim_range: Range of variation for latent components (e.g., [-3, 3]).
        original_min: Min value for reverse preprocessing.
        original_max: Max value for reverse preprocessing.
        original_shape: Shape of the input data.
        n_nodes: Number of nodes in the mesh.
        device: Torch device (e.g., 'cuda' or 'cpu').

    Returns:
        None
    """
    # Load the embeddings
    embeddings_df = pd.read_csv(embeddings_path)  # Assuming CSV format
    embeddings = torch.tensor(embeddings_df.iloc[:, 1:].values, dtype=torch.float32).to(device)  # Exclude ID column

    # Compute the mean embedding
    mean_embedding = embeddings.mean(dim=0).unsqueeze(0)  # Shape: [1, latent_dim]

    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    original_min = original_min.to(device)
    original_max = original_max.to(device)

    latent_dim = mean_embedding.shape[1]  # Number of latent dimensions

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Range of variation for each latent component
    # variation_values = torch.linspace(latent_dim_range[0], latent_dim_range[1], steps=6).to(device)  # E.g., 10 steps between -3 and +3
    variation_values = latent_dim_range

    # Vary each latent component one at a time
    for dim in range(latent_dim):
        # Create directory for this latent dimension
        dim_output_dir = os.path.join(output_dir, f"latent_dim_{dim}")
        os.makedirs(dim_output_dir, exist_ok=True)

        for value in variation_values:
            # Copy the mean embedding
            modified_embedding = mean_embedding.clone()

            # Vary the selected dimension
            modified_embedding[0, dim] = value

            # Generate the synthetic shape
            with torch.no_grad():
                decoded = model.decoder(modified_embedding)  # Decode the modified latent embedding
                decoded = decoded.view(1, original_shape[0], original_shape[1], original_shape[2], original_shape[3])  # Reshape to [batch_size, C, T, H, W]

                # Reverse preprocessing to get the final shape
                synthetic_shape = reverse_preprocessing(
                    decoded[0],  # batch size = 1
                    n_nodes,
                    original_shape,
                    original_min,
                    original_max,
                    normalize=True
                )

                # Save the synthetic shape
                save_as_vtk_emb(
                    synthetic_shape.cpu(), 
                    # f"latent_dim_{dim}_value_{value:.2f}", 
                    f"latent_dim_{dim}_value_{int(value)}", 
                    dim_output_dir
                )

    print("Synthetic shapes generated for varying latent components.")




from tqdm import tqdm

def vary_latent_components_and_generate_shapes2(
    model, model_path, embeddings_path, output_dir, latent_dim_range, 
    original_min, original_max, original_shape, n_nodes, device
):
    """
    Shapes are not denormalized
    Vary each latent component one at a time and generate synthetic shapes.

    Args:
        model: Trained autoencoder model.
        model_path: Path to the trained model checkpoint.
        embeddings_path: Path to the file containing precomputed embeddings.
        output_dir: Directory to save generated synthetic shapes.
        latent_dim_range: Range of variation for latent components (e.g., [-3, 3]).
        original_min: Min value for reverse preprocessing.
        original_max: Max value for reverse preprocessing.
        original_shape: Shape of the input data.
        n_nodes: Number of nodes in the mesh.
        device: Torch device (e.g., 'cuda' or 'cpu').

    Returns:
        None
    """
    # Load embeddings
    embeddings_df = pd.read_csv(embeddings_path)
    embeddings = torch.tensor(embeddings_df.iloc[:, 1:].values, dtype=torch.float32).to(device)

    # Compute the mean embedding
    mean_embedding = embeddings.mean(dim=0).unsqueeze(0)  # Shape: [1, latent_dim]

    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    original_min = original_min.to(device)
    original_max = original_max.to(device)

    latent_dim = mean_embedding.shape[1]  # Number of latent dimensions
    os.makedirs(output_dir, exist_ok=True)

    # Generate variation values if not already tensor
    if not isinstance(latent_dim_range, torch.Tensor):
        variation_values = torch.linspace(latent_dim_range[0], latent_dim_range[1], steps=10).to(device)
    else:
        variation_values = latent_dim_range

    # Process each latent dimension
    for dim in tqdm(range(latent_dim), desc="Processing Latent Dimensions"):
        dim_output_dir = os.path.join(output_dir, f"latent_dim_{dim}")
        os.makedirs(dim_output_dir, exist_ok=True)

        for value in variation_values:
            modified_embedding = mean_embedding.clone()
            modified_embedding[0, dim] = value

            try:
                # Decode and process the synthetic shape
                with torch.no_grad():
                    decoded = model.decoder(modified_embedding)
                    decoded = decoded.view(1, *original_shape)

                    synthetic_shape = reverse_preprocessing(
                        decoded[0], n_nodes, original_shape, original_min, original_max, normalize=False
                    )

                    save_as_vtk_emb(
                        synthetic_shape.cpu(), 
                        # decoded.cpu(), 
                        f"latent_dim_{dim}_value_{int(value)}", 
                        dim_output_dir
                    )
            except Exception as e:
                print(f"Error processing latent dimension {dim} value {value}: {e}")

    print("Synthetic shapes generated for varying latent components.")



import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def vary_TTN_comp_and_generate_shapes(
    model, model_path, embeddings_path, output_dir, original_min, 
    original_max, original_shape, n_nodes, device, affected_quantile="Q3"
):
    """
    Vary the latent C4 component while keeping other latent components fixed at their mean.
    Reverse preprocessing is applied before saving.

    Args:
        model: Trained autoencoder model.
        model_path: Path to the trained model checkpoint.
        motion_comp_ttn: DataFrame containing motion embeddings and TTN labels.
        output_dir: Directory to save generated synthetic shapes.
        original_min: Min value for reverse preprocessing.
        original_max: Max value for reverse preprocessing.
        original_shape: Shape of the input data.
        n_nodes: Number of nodes in the mesh.
        device: Torch device (e.g., 'cuda' or 'cpu').
        affected_quantile: The quantile group to use for TTN+ vs TTN- comparison.

    Returns:
        None
    """
    
    motion_comp_ttn = pd.read_csv(embeddings_path)
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    original_min = original_min.to(device)
    original_max = original_max.to(device)

    # Get all latent variables except C4
    latent_dims = [col for col in motion_comp_ttn.columns if col.startswith("c_") and col != "c_4"]

    # Compute mean values for all latents except C4
    mean_latents = motion_comp_ttn[latent_dims].mean().values

    # Select the affected quantile (e.g., Q3)
    ttn_positive = motion_comp_ttn[(motion_comp_ttn["TTN"] == 1) & (motion_comp_ttn["C4_quantile"] == affected_quantile)]
    ttn_negative = motion_comp_ttn[(motion_comp_ttn["TTN"] == 0) & (motion_comp_ttn["C4_quantile"] == affected_quantile)]

    # Display subject counts
    print(f"TTN+ subjects in {affected_quantile}: {len(ttn_positive)}")
    print(f"TTN- subjects in {affected_quantile}: {len(ttn_negative)}")

    # Extract mean embeddings for TTN+ and TTN-
    mean_ttn_positive = ttn_positive.iloc[:, 1:-2].mean().values  # Exclude IDs and labels
    mean_ttn_negative = ttn_negative.iloc[:, 1:-2].mean().values


    # Compute min/max C4 values for Q3
    q3_min = motion_comp_ttn[motion_comp_ttn["C4_quantile"] == affected_quantile]["c_4"].min()
    q3_max = motion_comp_ttn[motion_comp_ttn["C4_quantile"] == affected_quantile]["c_4"].max()

    # Define C4 range only within Q3
    c4_range = np.linspace(q3_min, q3_max, num=10)
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    for label, mean_embedding in zip(["TTN+", "TTN-"], [mean_ttn_positive, mean_ttn_negative]):
        latent_vectors = np.tile(mean_latents, (len(c4_range), 1))  # Replicate mean values
        latent_vectors = np.hstack((latent_vectors, c4_range.reshape(-1, 1)))  # Insert C4 values

        # Convert to tensor for model input
        latent_vectors = torch.tensor(latent_vectors).float().to(device)

        # Decode reconstructed heart motion
        with torch.no_grad():
            reconstructed_motions = model.decoder(latent_vectors).detach().cpu()
            print(reconstructed_motions.shape)
            

        # Apply reverse preprocessing before saving
        for i, motion in enumerate(reconstructed_motions):
            
            motion = motion.view(*original_shape)
            print(motion.shape)
            processed_shape = reverse_preprocessing(
                motion, n_nodes, original_shape, original_min, original_max, normalize=False
            )
            # Save in separate TTN+ and TTN- directories
            if label == "TTN+":
                save_as_vtk_emb(processed_shape, f"C4_{c4_range[i]:.2f}_TTN", output_dir)
            else:
                save_as_vtk_emb(processed_shape, f"C4_{c4_range[i]:.2f}NoTTN", output_dir)

            # save_as_vtk_emb(processed_shape, f"{label}_C4_{c4_range[i]:.2f}", output_dir)

        print(f"Generated synthetic shapes for {label} across C4 variations.")

    print("Shape generation completed.")



import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm

def vary_latent_components_and_generate_shapes_with_quantiles(
    model, model_path, embeddings_path, output_dir, quantile_ranges, 
    original_min, original_max, original_shape, n_nodes, device
):
    """
    Generate synthetic shapes based on quantiles of latent embeddings.

    Args:
        model: Trained autoencoder model.
        model_path: Path to the trained model checkpoint.
        embeddings_path: Path to the file containing precomputed embeddings.
        output_dir: Directory to save generated synthetic shapes.
        quantile_ranges: List of quantile thresholds (e.g., [0.01, 0.495, 0.99]).
        original_min: Min value for reverse preprocessing.
        original_max: Max value for reverse preprocessing.
        original_shape: Shape of the input data.
        n_nodes: Number of nodes in the mesh.
        device: Torch device (e.g., 'cuda' or 'cpu').

    Returns:
        None
    """
    # Load embeddings
    embeddings_df = pd.read_csv(embeddings_path)
    embeddings = torch.tensor(embeddings_df.iloc[:, 1:].values, dtype=torch.float32).to(device)

    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    original_min = original_min.to(device)
    original_max = original_max.to(device)

    latent_dim = embeddings.size(1)  # Number of latent dimensions
    os.makedirs(output_dir, exist_ok=True)

    # Process each latent dimension for quantile-based analysis
    for dim in tqdm(range(latent_dim), desc="Processing Latent Dimensions for Quantiles"):
        dim_output_dir = os.path.join(output_dir, f"latent_dim_{dim}")
        os.makedirs(dim_output_dir, exist_ok=True)

        # Extract the embeddings for the current dimension
        dim_values = embeddings[:, dim].cpu().numpy()

        # Compute quantiles for this dimension
        quantile_values = np.quantile(dim_values, quantile_ranges)

        for i in range(len(quantile_values) - 1):
            # Find the subjects within the quantile range
            lower_bound, upper_bound = quantile_values[i], quantile_values[i + 1]
            mask = (dim_values >= lower_bound) & (dim_values < upper_bound)
            selected_embeddings = embeddings[mask]

            if selected_embeddings.size(0) == 0:
                print(f"No embeddings found for latent dimension {dim} in range {lower_bound}-{upper_bound}")
                continue

            # Compute the mean embedding for this quantile range
            mean_embedding = selected_embeddings.mean(dim=0, keepdim=True)

            try:
                # Decode and process the synthetic shape
                with torch.no_grad():
                    decoded = model.decoder(mean_embedding)
                    decoded = decoded.view(1, *original_shape)

                    synthetic_shape = reverse_preprocessing(
                        decoded[0], n_nodes, original_shape, original_min, original_max, normalize=True
                    )

                    save_as_vtk_emb(
                        synthetic_shape.cpu(), 
                        f"latent_dim_{dim}_quantile_{i+1}", 
                        dim_output_dir
                    )
            except Exception as e:
                print(f"Error processing latent dimension {dim} quantile range {lower_bound}-{upper_bound}: {e}")

    print("Synthetic shapes generated for quantile-based latent analysis.")


def analyze_latent_variable_impact_with_quantiles(
    embeddings_path, output_dir, quantile_ranges, subject_ids_all, pos_all, 
    original_min, original_max, device
):
    """
    Analyze the impact of latent variables on LV morphology using quantiles, and visualize results.

    Args:
        embeddings_path: Path to the precomputed embeddings file.
        pickle_file_path: Path to the pickle file containing subject IDs and meshes.
        output_dir: Directory to save results.
        quantile_ranges: List of quantile ranges (e.g., [[0, 0.01], [0.095, 0.105], ...]).
        latent_dim_indices: List of latent dimensions to analyze (e.g., [0, 1, 2]).
        pos_all: Mesh positions for all subjects.
        original_min: Min value for reverse preprocessing.
        original_max: Max value for reverse preprocessing.
        device: Torch device (e.g., 'cuda' or 'cpu').

    Returns:
        None
    """
    # Load embeddings
    embeddings_df = pd.read_csv(embeddings_path)  # Assuming CSV format
    embeddings = embeddings_df.iloc[:, 1:].values  # Exclude subject IDs
    latent_dim = embeddings.shape[1]  # Number of latent dimensions

    # Convert original_min and original_max to NumPy (if selected_meshes is a NumPy array)
    original_min = original_min.cpu().numpy()  # Convert to NumPy array
    original_max = original_max.cpu().numpy()  # Convert to NumPy array


    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each latent dimension to analyze
    for dim in tqdm(range(latent_dim), desc="Processing Latent Dimensions for Quantiles"):
        dim_output_dir = os.path.join(output_dir, f"latent_dim_{dim}")
        os.makedirs(dim_output_dir, exist_ok=True)

        # Extract latent values for the current dimension
        latent_values = embeddings[:, dim]

        # Compute quantile thresholds for the dimension
        quantile_thresholds = [np.quantile(latent_values, q) for q in np.array(quantile_ranges).flatten()]

        # Prepare visualization figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Latent Dimension {dim}: Average Shapes Across Quantiles")

        # Process each quantile range
        for i in range(0, len(quantile_thresholds), 2):
            lower_bound, upper_bound = quantile_thresholds[i], quantile_thresholds[i + 1]

            # Select subjects in the current quantile range
            subject_indices = np.where((latent_values >= lower_bound) & (latent_values < upper_bound))[0]
            selected_meshes = pos_all[subject_indices]  # Extract meshes for selected subjects
            # selected_subjects = subject_ids_all[subject_indices]

            if len(selected_meshes) == 0:
                print(f"No subjects found in latent dimension {dim} range {lower_bound}-{upper_bound}.")
                continue

            # Unscale meshes (reverse preprocessing)
            unscaled_meshes = (selected_meshes - original_min) / (original_max - original_min)

            # Compute the average shape for the current quantile range
            average_mesh = unscaled_meshes.mean(axis=0)

            # Rescale the averaged mesh
            rescaled_mesh = average_mesh * (original_max - original_min) + original_min

            # Save the averaged shape
            quantile_folder = os.path.join(dim_output_dir, f"quantile_{i//2 + 1}")
            os.makedirs(quantile_folder, exist_ok=True)
            save_as_vtk_emb(torch.tensor(rescaled_mesh), f"quantile_{i//2 + 1}", quantile_folder)

            # Visualize the average shape (e.g., first frame)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(rescaled_mesh[0, :, 0], rescaled_mesh[0, :, 1], rescaled_mesh[0, :, 2], s=2)
            ax.set_title(f"Latent Dim {dim} Quantile {i//2 + 1} - Frame 0")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.savefig(os.path.join(quantile_folder, f"quantile_{i//2 + 1}_frame_0.png"))
            plt.close(fig)

            print(f"Saved results for latent dimension {dim}, quantile range {lower_bound}-{upper_bound}.")

    print("Quantile-based latent variable analysis completed for test data.")


    # # Unique random seed per trial
    # seed = [1701, 1993, 2023, 42, 1994][trial.number % 5]  # cycle through predefined seeds
    # print(f"Using seed: {seed} for trial {trial.number}")
    
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# def optuna_objective(trial):
#     """Optuna objective function for hyperparameter tuning."""
#     # Sample hyperparameters
#     # alpha = trial.suggest_float("alpha", 0.0001, 1.0, step=0.01)  # Range for alpha
#     # gamma = trial.suggest_float("gamma", 0.0001, 1.0, step=0.01)  # Range for alpha
#     # beta = trial.suggest_float("beta", 0.0001, 1.0, step=0.001)  # Range for alpha
#     # beta = trial.suggest_float("beta", 3e-5, 1e-4, log=True)  # Log scale for beta (small values)
#     # beta = trial.suggest_float("beta", 1e-6, 1e-2, log=True)
#     # beta = 0
    
#     alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
    # gamma = trial.suggest_float("gamma", 1e-5, 1.0, log=True)
    # beta = trial.suggest_float("beta", 1e-5, 1.0, log=True)
    # delta = trial.suggest_float("delta", 1e-5, 1.0, log=True)

    # encoded_dim = trial.suggest_categorical("encoded_dim", [16, 32, 64, 128])

    



    # # Update args with sampled values
    # args.alpha = alpha
    # args.beta = beta
    # args.gamma = gamma
    # args.delta = delta
    # args.encoded_dim = encoded_dim 

    # # --- 1. Random seed (for weight init + split control)
    # # seed_list = [1701, 1993, 2023, 42, 1994]
    # # seed = seed_list[trial.number % len(seed_list)]  # cycle through
    # # trial.set_user_attr("seed", seed)

    # # import random, numpy as np, torch
    # # random.seed(seed)
    # # np.random.seed(seed)
    # # torch.manual_seed(seed)
    # # torch.cuda.manual_seed_all(seed)
    # # torch.backends.cudnn.deterministic = True
    # # torch.backends.cudnn.benchmark = False


    # ### Dynamically create output directory for each trial
    
    # # optuna_out_dir = os.path.join(args.output_base_dir, f"optuna_trial_{trial.number}_latent{encoded_dim}_alpha{alpha}_beta{beta}")
    # # os.makedirs(optuna_out_dir, exist_ok=True)

    # # out_dir = os.path.join(args.output_base_dir, f"emb_deepmesh_ae_optuna_seed")

    # optuna_out_dir = os.path.join(args.output_base_dir, f"optuna_trial_{trial.number}")
    # os.makedirs(optuna_out_dir, exist_ok=True)

    # # Initialize model
    # # model = BetaVAE(encoded_dim=args.encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    # # model = I3DAutoencoder(encoded_dim=args.encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    # # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # # Initialize model
    # model = BetaVAE(encoded_dim=encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # trial_start = time.time()

    # # Train the model (few epochs for speed)
    # # val_losses = train_betavae_smooth(model, f"optuna_trial_{trial.number}", train_loader, val_loader, optimizer, optuna_out_dir, n_frames, n_nodes, args)
    # # val_losses , val_recon_losses =  train_AE_smooth(model, f"optuna_trial_{trial.number}", train_loader, val_loader, optimizer, optuna_out_dir, n_frames, n_nodes, args)
    # # val_losses , val_recon_losses =  train_betavae_smooth(model, f"optuna_trial_{trial.number}", train_loader, val_loader, optimizer, optuna_out_dir, n_frames, n_nodes, args)
    # val_losses , val_recon_losses, val_kl_losses, val_lap_losses, val_norm_losses, val_t_losses = train_betavae_smooth(model, f"optuna_trial_{trial.number}", train_loader, val_loader, optimizer, optuna_out_dir, n_frames, n_nodes, args)
    
    
    # trial_end = time.time()
    # trial_duration = trial_end - trial_start
    # print(f"Trial {trial.number} took {trial_duration:.2f} seconds")
    
    # # Get final validation loss (last recorded value)
    # final_val_loss = val_losses[-1]  # Use last validation loss
    
    # final_recon_loss = val_recon_losses[-1]
    # # Store it as user attribute
    # trial.set_user_attr("recon_loss", final_recon_loss)


    # log_row = {
    # "trial_number": trial.number,
    # "trial_duration_sec": trial_duration,
    # "alpha": alpha,
    # "beta": beta,
    # "gamma": gamma,
    # "delta" : delta,
    # "encoded_dim": encoded_dim,
    # "val_loss": val_losses[-1],
    # "val_recon_loss": val_recon_losses[-1],
    # "val_kl_loss": val_kl_losses[-1],
    # "val_lap_loss": val_lap_losses[-1],
    # "val_norm_loss": val_norm_losses[-1],
    # "val_temporal_loss": val_t_losses[-1],
    # }
    
    # # JSON version (optional)
    # with open(os.path.join(optuna_out_dir, f"trial_{trial.number}.json"), "w") as f:
    #     json.dump(log_row, f, indent=4)
        


    # return final_val_loss  # Optuna minimizes this


# # Argument Parsing
# def parse_args():
#     parser = argparse.ArgumentParser(description="Train and test FactorVAE.")
#     parser.add_argument("--device", type=str, default="cuda:4", help="Device (e.g., 'cuda:1').")
#     parser.add_argument("--encoded_dim", type=int, default=32, help="Latent space dimension.") ##default=128   16
#     parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
#     parser.add_argument("--gamma", type=float, default=1, help="Gamma for Normal loss.") ## default=10.0
#     parser.add_argument("--alpha", type=float, default=1, help="Alpha for total smoothing loss.") ## default=10.0
#     parser.add_argument("--beta", type=float, default= 0.01, help="Beat for KL loss.")  ## default=1.0   0.001
#     parser.add_argument("--input_dir", type=str, default="/cardiac/sk/SpaceMAP_deepmesh", help="Input directory.")
#     parser.add_argument("--output_base_dir", type=str, default="/skalaie/motion_code/DR_motion/", help="Output directory.")
#     parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
#     parser.add_argument("--atlas", type=str, default="/ukb/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec/avg/LVmyo_mesh_avg_decim_0.95_0.vtk", help="Atlas dir.")

    
#     return parser.parse_args()


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train and test FactorVAE.")
    parser.add_argument("--device", type=str, default="cuda:4", help="Device (e.g., 'cuda:4').")    
    # parser.add_argument("--exp_name", type=str, default="VAE_r2plus1d_18", help="Name of the method")## SPTVAE
    parser.add_argument("--exp_name", type=str, default="VAE_r3d_18", help="Name of the method")## SPTVAE
    parser.add_argument("--encoded_dim", type=int, default=32, help="Latent space dimension.") ##default=128   16
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--lr", type=int, default=1e-4, help="Learning Rate.") ##1e-4
    parser.add_argument("--alpha", type=float, default=0.001, help="Alpha for total smoothing laplacian loss.") ## default=0.01
    parser.add_argument("--beta", type=float, default= 2e-05, help="Beta for KL loss.")  ## default=1.0   0.001
    parser.add_argument("--gamma", type=float, default=0.001, help="Gamma for normal loss.") ## default=0.01
    parser.add_argument("--delta", type=float, default=0.0001, help="temporal_smoothness")  ## default=1.0   0.001
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="L2 regularization weight.")
    parser.add_argument("--input_dir", type=str, default="/ukb/sk/packdata_deepmesh/", help="Input directory.") 
    parser.add_argument("--output_base_dir", type=str, default="/skalaie/motion_code/DR_motion/SPTVAE/results_VAE_optuna_10k/VAE_32_1/", help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--atlas", type=str, default="/ukb/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec_quadsim/avg/LVmyo_mesh_avg_decim_1200_0_quadsim.vtk", help="Atlas dir.")
    
    return parser.parse_args()

def _init_():
    if not os.path.exists(args.output_base_dir):
        os.makedirs(args.output_base_dir)
            
    checkpoints_dir = os.path.join(args.output_base_dir, 'checkpoints')    
    if not os.path.exists(args.output_base_dir +'checkpoints/'):
        os.makedirs(args.output_base_dir +'checkpoints/')
# python main_finetune_motion_factorVAE_new.py --latent_dim 64 --epochs 20 --gamma 5.0 

if __name__ == "__main__":
    
    args = parse_args()
    _init_()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Dynamically create output directory based on encoded_dim
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    

    print('Beta:',args.beta)
    print('Alpha:',args.alpha)
    print('gamma:',args.gamma)    
    print('delta:',args.delta)


    print('LR:',args.lr)
    ############  Test dummy data  ######################
    # N, T, Nodes, C = 100, 25, 1187, 3  # Example dimensions
    # pos = np.random.rand(N, T, Nodes, C)
    # n_frames = T
    # n_nodes = Nodes
    # H = int(math.sqrt(Nodes))
    # W = math.ceil(Nodes / H)
    # subject_ids = list(range(N))
    # train_size = int(0.9 * len(pos))
    # train_pos, val_pos = pos[:train_size], pos[train_size:]
    # train_ids, val_ids = subject_ids[:train_size], subject_ids[train_size:]
    # test_pos = pos
    # test_ids = subject_ids 

    ########## DeepMesh #######################################################################
    # NUM_SUBJECTS = 37039 ## with outliers
    NUM_SUBJECTS = 36856
    print('NUM_SUBJECTS',NUM_SUBJECTS)
    # INPUT_DIR = "/cardiac/sk/SpaceMAP_deepmesh"
    # pickle_file_path = f'{INPUT_DIR}/id_pos_NS{NUM_SUBJECTS}_parallel_filtered_outliers.pkl'
    # pickle_file_path = f'{args.input_dir}/id_pos_NS{NUM_SUBJECTS}_parallel_filtered_outliers.pkl'
    pickle_file_path = f'{args.input_dir}/id_pos_NS{NUM_SUBJECTS}_FRstep_1_parallel_filtered_outliers_quadsim.pkl' ## 50 frames

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
    
    # # # Real data  #####
    # with open(pickle_file_path, "rb") as f:
    #     subject_ids_all, pos_all = pickle.load(f)


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
        
    
    # data = torch.load("subjects_2000_quadsim.pt")
    # # data = torch.load("subjects_1000.pt")

    # pos_all = data['pos']
    # subject_ids_all = data['subject_ids']
    # pos = pos_all
    # subject_ids = subject_ids_all
    
    
    
    # data = torch.load("/skalaie/motion_code/DR_motion/subjects_20k_quadsim.pt")
    data = torch.load("/skalaie/motion_code/DR_motion/subjects_10k_quadsim.pt")
    
    pos_all = data['pos']
    subject_ids_all = data['subject_ids']


    
    N, n_frames, n_nodes, C = pos_all.shape   # pos.shape : [N,T,nodes,3]
    # H = int(math.sqrt(n_nodes))
    # W = math.ceil(n_nodes / H)
    H = 30
    W = math.ceil(n_nodes / H)
    print( "Number of Nodes", n_nodes)
    print( "Number of frames", n_frames)
    
    ##################################################################################################
    ### Split data
    # pos = pos_all
    # subject_ids = subject_ids_all
    # train_size = int(0.95 * len(pos))
    # train_pos, val_pos = pos[:train_size], pos[train_size:]
    # train_ids, val_ids = subject_ids[:train_size], subject_ids[train_size:]

    # ### Whole population 
    # test_pos = pos_all
    # test_ids = subject_ids_all 

    # # test_pos = val_pos
    # # test_ids = val_ids


    # # # # Datasets and loaders
    # train_dataset = MotionDataset(train_pos, train_ids)
    # val_dataset = MotionDataset(val_pos, val_ids)
    # test_dataset = MotionDataset(test_pos, test_ids)
    
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # test_loader =DataLoader(test_dataset, batch_size=1, shuffle=False)
    ################################################################################################

    # Convert to tensors
    # subject_ids_all = np.array(subject_ids_all)
    # pos_all = np.array(pos_all)
    assert len(pos_all) == len(subject_ids_all)

    # Wrap everything into a dataset first
    full_dataset = MotionDataset(pos_all, subject_ids_all)

    # Define split ratios
    total_samples = len(full_dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    # Use random_split (with reproducibility)
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    #########################################################################################################
    # Extract IDs from each subset
    train_ids = [train_dataset.dataset.subject_ids[i] for i in train_dataset.indices]
    val_ids = [val_dataset.dataset.subject_ids[i] for i in val_dataset.indices]
    test_ids = [test_dataset.dataset.subject_ids[i] for i in test_dataset.indices]
    
    df = pd.DataFrame({"eid_18545": train_ids})
    df.to_csv(f"{args.output_base_dir}/train_ids.csv", index=False)
    
    df = pd.DataFrame({"eid_18545": val_ids})
    df.to_csv(f"{args.output_base_dir}/val_ids.csv", index=False)
    
    df = pd.DataFrame({"eid_18545": test_ids})
    df.to_csv(f"{args.output_base_dir}/test_ids.csv", index=False)
    
    
    # Extract  positions using indices from the full dataset
    train_pos = [train_dataset.dataset.data[i] for i in train_dataset.indices]
    val_pos = [val_dataset.dataset.data[i] for i in val_dataset.indices]
    test_pos = [test_dataset.dataset.data[i] for i in test_dataset.indices]

    # # Convert to tensor (optional, for saving)
    # train_pos_tensor = torch.stack(train_pos)  # Shape: [num_test_samples, n_frames, n_nodes, 3]
    # train_ids_tensor = torch.tensor(train_ids)  # Convert list of IDs to tensor
    
    # val_pos_tensor = torch.stack(val_pos)  # Shape: [num_test_samples, n_frames, n_nodes, 3]
    # val_ids_tensor = torch.tensor(val_ids)  # Convert list of IDs to tensor
    
    # test_pos_tensor = torch.stack(test_pos)  # Shape: [num_test_samples, n_frames, n_nodes, 3]
    # test_ids_tensor = torch.tensor(test_ids)  # Convert list of IDs to tensor

    # # Save to file
    # save_path = os.path.join(args.output_base_dir, "train_pos_and_ids.pt")
    # torch.save({
    #     "pos": train_pos_tensor,
    #     "subject_ids": train_ids_tensor
    # }, save_path, pickle_protocol=4)

    # save_path = os.path.join(args.output_base_dir, "val_pos_and_ids.pt")
    # torch.save({
    #     "pos": val_pos_tensor,
    #     "subject_ids": val_ids_tensor
    # }, save_path, pickle_protocol=4)
    
    # save_path = os.path.join(args.output_base_dir, "test_pos_and_ids.pt")
    # torch.save({
    #     "pos": test_pos_tensor,
    #     "subject_ids": test_ids_tensor
    # }, save_path, pickle_protocol=4)
    
    train_pos_np = np.array(train_pos)            # Shape: [N_train, T, nodes, 3]
    train_ids_np = np.array(train_ids)            # Shape: [N_train]

    val_pos_np = np.array(val_pos)
    val_ids_np = np.array(val_ids)

    test_pos_np = np.array(test_pos)
    test_ids_np = np.array(test_ids)

    torch.save({'pos': train_pos_np, 'subject_ids': train_ids_np}, f"{args.output_base_dir}/train_data.pt", pickle_protocol=4)
    torch.save({'pos': val_pos_np, 'subject_ids': val_ids_np}, f"{args.output_base_dir}/val_data.pt", pickle_protocol=4)
    torch.save({'pos': test_pos_np, 'subject_ids': test_ids_np}, f"{args.output_base_dir}/test_data.pt", pickle_protocol=4)

    print(f"Saved train/val/test data!")
    ################################################################################################
    # model = Cardio4DVAE(encoded_dim=args.encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    # model = I3DAutoencoder(encoded_dim=args.encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    # optimizer_vae = torch.optim.Adam(model.parameters(), lr=1e-4)
    # model = AE(encoded_dim=args.encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    
    model = Cardio4DVAE(encoded_dim=args.encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    ###############################################################################################
    ## Train
    ###############################################################################################
    # modelName = "r3d_18_betavae_resEncfcDec"
    # modelName = "r3d_18_bvae_L2smooth"


    
    # train_betavae(
    #     model, modelName, train_loader, val_loader, optimizer_vae, out_dir, n_frames, n_nodes, args
    # )

    # train_betavae_smooth(
    #     model, modelName, train_loader, val_loader, optimizer_vae, out_dir, n_frames, n_nodes, args
    #     )


    
    
    # Initialize model
    model = Cardio4DVAE(encoded_dim=args.encoded_dim, input_shape=(3, n_frames, H, W)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    trial_start = time.time()

    # Train the model (few epochs for speed)
    # val_losses = train_betavae_smooth(model, f"optuna_trial_{trial.number}", train_loader, val_loader, optimizer, optuna_out_dir, n_frames, n_nodes, args)
    # val_losses , val_recon_losses =  train_AE_smooth(model, f"optuna_trial_{trial.number}", train_loader, val_loader, optimizer, optuna_out_dir, n_frames, n_nodes, args)
    # val_losses , val_recon_losses =  train_betavae_smooth(model, f"optuna_trial_{trial.number}", train_loader, val_loader, optimizer, optuna_out_dir, n_frames, n_nodes, args)
    val_losses , val_recon_losses, val_kl_losses, val_lap_losses, val_norm_losses, val_t_losses = train_betavae_smooth(model, args.exp_name, train_loader, val_loader, optimizer, args.output_base_dir, n_frames, n_nodes, args)
    
    
    trial_end = time.time()
    trial_duration = trial_end - trial_start
    print(f"Training {args.exp_name} took {trial_duration:.2f} seconds")
    ###############################################################################################
    ## Test
    ###############################################################################################
    # ## modelName = "r3d_18_betavae_resEncfcDec_epoch_20"
    # model_path = os.path.join(out_dir, modelName) 
    # # output_name = f"LVmyo_motion_embeddings_mu_{args.encoded_dim}_NS_{NUM_SUBJECTS}.csv"
    # output_name = f"LVmyo_motion_embeddings_{args.encoded_dim}_NS_{NUM_SUBJECTS}.csv"
    

    # output_file = os.path.join(out_dir, output_name)
    # test_model_and_save_embeddings(model, model_path, test_loader, output_file,MEAN='mu')
  

    ###############################################################################################
    ## Embedding
    ###############################################################################################
    # n_frames = 25 
    # H = 34
    # W = 35
    # n_nodes = 1187

    # original_min =  torch.tensor([-85.5211])
    # original_max =  torch.tensor([55.7752])
    # # original_shape = torch.Size([3, n_frames, H, W])
    # original_shape = (3, n_frames, H, W)
    
    # ##### Leave one out embeddings
    embeddings_path =f"{args.output_base_dir}/LVmyo_motion_embeddings_{args.encoded_dim}_NS_{NUM_SUBJECTS}.csv"  # Precomputed embeddings
    
    # embeddings_path =f"{out_dir}/LVmyo_motion_embeddings_mu_{args.encoded_dim}_NS_{NUM_SUBJECTS}.csv"  # Precomputed embeddings
    # # output_dir = "./embedding_reconstructed_shapes_30"
    # # selected_dims = [27, 22, 1]  # Embeddings to use
    # # selected_dims = [27, 12, 6, 9]    
    # # selected_dims = [22, 3 , 10]
    # # selected_dims = [30] # Embeddings to use
    # selected_dims = [15, 13, 1]  # Top embedding dimensions to analyze



    ##### Embedding Importance   
    # # Usage
    # # Criterion (e.g., MSE)
    # criterion = nn.MSELoss()
    # output_file = f"{out_dir}/embedding_importance_scores_{NUM_SUBJECTS}_outlierfiltered.csv"
    # # Calculate embedding importance
    # importance_scores = calculate_embedding_importance(model, model_path, test_loader, criterion, output_file , device)
    # # Visualize the importance scores
    # figure_path = f"{out_dir}/embedding_importance_scores_{NUM_SUBJECTS}_outlierfiltered.png"
    # importance_scores = np.ravel(importance_scores)  # Convert to 1D array if necessary
    ######################################################################################
    # reconstruct_shapes_from_embeddings(model, model_path, embeddings_path, output_dir, selected_dims,test_loader, device)
    
    # reconstruct_shapes_from_embeddings_(model, model_path, embeddings_path, out_dir, selected_dims, original_min, original_max, original_shape,n_nodes, device)
    
    # output_dir = f"{out_dir}/population_mean_reconstructed_shapes"
    # reconstruct_with_mean_and_3sd(
    # model,
    # model_path,
    # embeddings_path,
    # output_dir,  
    # selected_dims,
    # original_min,
    # original_max,
    # original_shape,
    # n_nodes,
    # device
    # )
    
    ############## shapes with latent ###################################################
    # output_dir = f"{out_dir}/synthetic_shapes2_norm"
    # # latent_dim_range=(-3, 3)
    # # latent_dim_range=torch.linspace(-3, 3,steps=6)
    
    # # latent_dim_range = torch.arange(-3, 4, step=1).to(device)
    # latent_dim_range = torch.arange(-10, 11, step=2).to(device)

    # vary_latent_components_and_generate_shapes(
    # model,
    # model_path,
    # embeddings_path,
    # output_dir,
    # latent_dim_range,
    # original_min,
    # original_max,
    # original_shape,
    # n_nodes,
    # device
    # )
    

    # vary_latent_components_and_generate_shapes2(
    # model,
    # model_path,
    # embeddings_path,
    # output_dir,
    # latent_dim_range,
    # original_min,
    # original_max,
    # original_shape,
    # n_nodes,
    # device
    # )
    
    # embeddings_path =f"{out_dir}/LVmyo_motion_embeddings_{args.encoded_dim}_NS_{NUM_SUBJECTS}_ttn.csv"  # Precomputed embeddings

    # output_dir = f"{out_dir}/synthetic_TTN_Q3"

    # vary_TTN_comp_and_generate_shapes(
    # model, model_path, embeddings_path, output_dir, original_min, 
    # original_max, original_shape, n_nodes, device, affected_quantile="Q3")
    #### Quantiles ##############################################
    
    # output_dir = f"{out_dir}/synthetic_shapes_quantile"

    # quantile_ranges = [0.01, 0.095, 0.495, 0.895, 0.99]  # Define quantiles
    # # Run quantile-based shape generation
    # vary_latent_components_and_generate_shapes_with_quantiles(
    # model, model_path, embeddings_path, output_dir, quantile_ranges,
    # original_min, original_max, original_shape, n_nodes, device
    # )
    
    
    
    # Example Usage
    # averaged_shapes_dir = f"{out_dir}/averaged_shapes_quantile_20"
    # # latent_dim_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # Latent dimensions to visualize
    # # quantile_ranges = [[0, 0.01], [0.095, 0.105], [0.495, 0.505], [0.895, 0.905], [0.99, 1]]  # Quantile ranges
    
    # quantile_ranges = [[0, 0.05], [0.05, 0.25], [0.25, 0.75], [0.75, 0.95], [0.95, 1]]

    # output_dir = f"{out_dir}/averaged_shapes_quantile_20"  # Directory to save visualization figures
    
    
    # # Run Analysis
    # # analyze_latent_variable_impact_with_quantiles(
    # #     embeddings_path, test_loader, output_dir, quantile_ranges, 
    # #     original_shape, n_nodes, device
    # # )
    
    # analyze_latent_variable_impact_with_quantiles(
    # embeddings_path, output_dir, quantile_ranges, subject_ids_all, pos_all, 
    # original_min, original_max, device)