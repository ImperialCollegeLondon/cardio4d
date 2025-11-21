
import pyvista as pv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
import pickle
# The above code is importing the pandas library in Python using the `import` statement and aliasing
# it as `pd`. This allows the code to use the functionalities provided by the pandas library for data
# manipulation and analysis.
import pandas as pd
import os
import seaborn as sns
import numpy as np
import math
from tqdm import tqdm
import argparse
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance as chamfer_dis
from pytorch3d.loss import mesh_normal_consistency as normal_dis
from pytorch3d.loss import mesh_laplacian_smoothing as lap_dis

import torch.nn.functional as F

### global normalization
class MotionDataset(Dataset):
    def __init__(self, data, subject_ids):
        """
        Args:
            data: List or numpy array of shape [NUM_SUBJECTS, T, Nodes, C]
            subject_ids: List of subject IDs corresponding to the data
        """
        self.data = data
        self.subject_ids = subject_ids


        ## Normalizing each sample using the dataset-wide min and max values to preserve relative sizes.
        
        ## Calculate global min and max for the dataset
        all_data = np.concatenate(data, axis=0)  # Combine along the first axis
        self.global_min = all_data.min()
        self.global_max = all_data.max()
        
        # all_data = np.concatenate(data[0:5000], axis=0)  # Combine along the first axis
        # self.global_min = all_data.min()
        # self.global_max = all_data.max()

        # self.global_min = np.array([-85.5211])
        # self.global_max = np.array([55.7752])
        
        
        

        
        # self.global_min =  torch.tensor([-85.5211])
        # self.global_max =  torch.tensor([55.7752])
        
        # self.global_min = self.global_min.to(dtype=torch.float16)
        # self.global_max = self.global_max.to(dtype=torch.float16)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get a single sample and its ID
        sample = self.data[idx]  # Shape: [T, Nodes, C]
        subject_id = self.subject_ids[idx]

        # Normalize using global min and max
        sample = (sample - self.global_min) / (self.global_max - self.global_min)

        # Dynamically determine grid dimensions
        T, Nodes, C = sample.shape
        # H = int(math.sqrt(Nodes))
        H=30
        W = math.ceil(Nodes / H)
        padded_nodes = H * W

        # Pad if necessary
        if Nodes < padded_nodes:
                padding = padded_nodes - Nodes
                # sample = torch.nn.functional.pad(
                #     sample.clone().detach(),  # Safely create a copy
                #     (0, 0, 0, padding)
                # )
                sample = torch.nn.functional.pad(
                    torch.tensor(sample), (0, 0, 0, padding)
                )  # Resulting shape: [T, padded_nodes, C]

        # Convert to PyTorch tensor
            # Convert to tensor if not already a tensor
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)

        # sample = torch.tensor(sample)
        # sample = sample.clone().detach()  # Final tensor for reshaping
        # Reshape nodes into a grid [T, H, W, C] and permute to [C, T, H, W]
        sample = sample.view(T, H, W, C)
        sample = sample.permute(3, 0, 1, 2)  # Final shape: [C, T, H, W]

        return sample.float(), subject_id, self.global_min, self.global_max



from sklearn.decomposition import PCA

def get_node_to_grid_map(node_positions, H, W):
    # node_positions: [N, 3] array of mesh vertices
    pca = PCA(n_components=2)
    uv = pca.fit_transform(node_positions)  # [N, 2]
    
    # Normalize to [0, 1]
    uv = (uv - uv.min(0)) / (uv.max(0) - uv.min(0) + 1e-8)
    
    # Map to grid
    i_coords = (uv[:, 0] * (H - 1)).astype(int)
    j_coords = (uv[:, 1] * (W - 1)).astype(int)
    
    grid = -1 * np.ones((H, W), dtype=int)
    for node_idx, (i, j) in enumerate(zip(i_coords, j_coords)):
        grid[i, j] = node_idx  # mark where this node goes
    
    return grid  # grid[i, j] = node index (or -1 if empty)


 #########################################################################################################################

def reverse_preprocessing(decoded, n_nodes, original_shape, original_min, original_max, normalize=True):
    """
    Reverse the preprocessing steps to restore original shape and values.

    Args:
        decoded: Decoded tensor of shape [C, T, H, W].
        original_shape: Tuple of the original shape (T, Nodes, C).
        original_min: Minimum value of the original data.
        original_max: Maximum value of the original data.
        normalize: Whether normalization was applied during preprocessing.

    Returns:
        Tensor of shape [T, Nodes, C] with original values restored.
    """
    C, T, H, W = original_shape
    # C, T, H, W = decoded.shape
    
    # n_nodes = 1187
    # Reshape back to [T, Nodes, C]
    decoded = decoded.permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]
    decoded = decoded.view(T, -1, C)       # Flatten grid back to nodes
    # decoded = decoded.view(T, H * W, C)  # Flatten grid
    decoded = decoded[:, :n_nodes, :]        # Remove padding (if any)

    # # Undo normalization
    if normalize == True:
        # print('Denormalisation is Done')
        decoded = decoded * (original_max - original_min) + original_min

    return decoded


def save_as_vtk(reconstructed, original, subject_id, output_dir, args):
    """
    Save reconstructed data as a VTK mesh file.
    Args:
        reconstructed: Tensor of shape [50, 1406, 3] (frames, nodes, coordinates).
        original: Tensor of shape [50, 1406, 3].
        subject_id: Unique identifier for the subject.
        output_dir: Directory to save the VTK files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure compatibility
    assert reconstructed.shape == original.shape, "Reconstructed and original shapes must match"
    
    # atlas = pv.read('/ukb/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec/avg/LVmyo_mesh_avg_decim_0.95_0.vtk')
    atlas = pv.read(args.atlas)

    # Save each frame
    for frame_idx in range(reconstructed.shape[0]):
        frame_data = reconstructed[frame_idx].numpy()
        # mesh = pv.PolyData(frame_data)
        mesh = pv.PolyData(frame_data, atlas.faces)
        folder = f"{output_dir}/vtks/{subject_id}"
        os.makedirs(folder, exist_ok=True)
        mesh.save(f"{folder}/LVmyo_fr{frame_idx:02d}.vtk")
        
        ##### original ######################################
        frame_data_org = original[frame_idx]  # Shape [1406, 3]
        # Create PyVista PolyData
        points_org = frame_data_org.numpy()  # Convert to NumPy array
        # mesh_org = pv.PolyData(points_org)
        mesh_org = pv.PolyData(points_org, atlas.faces)
        
        folder_org = f'{output_dir}/vtks_org/{subject_id}'
        os.makedirs(folder_org, exist_ok=True)
        
        fname__org = f'{folder_org}/LVmyo_fr{frame_idx:02d}.vtk'  # Include frame index in the filename
        mesh_org.save(fname__org)


def save_as_vtk_emb(reconstructed, subject_id, output_dir, args):
    """
    Save reconstructed data as a VTK mesh file.
    Args:
        reconstructed: Tensor of shape [50, 1406, 3] (frames, nodes, coordinates).
        original: Tensor of shape [50, 1406, 3].
        subject_id: Unique identifier for the subject.
        output_dir: Directory to save the VTK files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure compatibility
    # assert reconstructed.shape == original.shape, "Reconstructed and original shapes must match"
    
    # atlas = pv.read('/ukb/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec/avg/LVmyo_mesh_avg_decim_0.95_0.vtk')
    atlas = pv.read(args.atlas)

    # Save each frame
    for frame_idx in range(reconstructed.shape[0]):
        frame_data = reconstructed[frame_idx].numpy()
        # mesh = pv.PolyData(frame_data)
        mesh = pv.PolyData(frame_data, atlas.faces)
        
        
        
        
        folder = f"{output_dir}/vtks/{subject_id}"        
        # folder = f"{output_dir}/vtks/"
        os.makedirs(folder, exist_ok=True)
        mesh.save(f"{folder}/LVmyo_fr{frame_idx:02d}.vtk")
        
        # ##### original ######################################
        # frame_data_org = original[frame_idx]  # Shape [1406, 3]
        # # Create PyVista PolyData
        # points_org = frame_data_org.numpy()  # Convert to NumPy array
        # mesh_org = pv.PolyData(points_org)
        
        # folder_org = f'{output_dir}/vtks_org/{subject_id}'
        # os.makedirs(folder_org, exist_ok=True)
        
        # fname__org = f'{folder_org}/LVmyo_fr{frame_idx:02d}.vtk'  # Include frame index in the filename
        # mesh_org.save(fname__org)



def save_as_vtk_emb_smooth(reconstructed, subject_id, output_dir, args, smooth=None):
    """
    Save reconstructed data as a VTK mesh file.
    Args:
        reconstructed: Tensor of shape [50, 1406, 3] (frames, nodes, coordinates).
        original: Tensor of shape [50, 1406, 3].
        subject_id: Unique identifier for the subject.
        output_dir: Directory to save the VTK files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure compatibility
    # assert reconstructed.shape == original.shape, "Reconstructed and original shapes must match"
    
    # atlas = pv.read('/ukb/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec/avg/LVmyo_mesh_avg_decim_0.95_0.vtk')
    atlas = pv.read(args.atlas)

    # Save each frame
    for frame_idx in range(reconstructed.shape[0]):
        frame_data = reconstructed[frame_idx].numpy()
        # mesh = pv.PolyData(frame_data)
        mesh = pv.PolyData(frame_data, atlas.faces)
        
        
        
        
        folder = f"{output_dir}/vtks/{subject_id}"        
        # folder = f"{output_dir}/vtks/"
        os.makedirs(folder, exist_ok=True)
        mesh.save(f"{folder}/LVmyo_fr{frame_idx:02d}.vtk")
        
        if smooth==True:
            # print('smoothing')
            mesh = mesh.smooth(n_iter=100)  # Apply Laplacian smoothing with 30 iterations

        
        # ##### original ######################################
        # frame_data_org = original[frame_idx]  # Shape [1406, 3]
        # # Create PyVista PolyData
        # points_org = frame_data_org.numpy()  # Convert to NumPy array
        # mesh_org = pv.PolyData(points_org)
        
        # folder_org = f'{output_dir}/vtks_org/{subject_id}'
        # os.makedirs(folder_org, exist_ok=True)
        
        # fname__org = f'{folder_org}/LVmyo_fr{frame_idx:02d}.vtk'  # Include frame index in the filename
        # mesh_org.save(fname__org)
        
        
        




def compute_total_correlation(z_sample, mu, log_var):
    """
    Compute Total Correlation (TC) loss.
    Uses Monte Carlo estimation of KL divergence between joint posterior and product of marginals.
    """
    batch_size, latent_dim = z_sample.shape

    # Compute log q(z|x) using diagonal Gaussian assumption
    log_qz_cond_x = -0.5 * (log_var + ((z_sample - mu) ** 2) / log_var.exp())  # Shape: [batch_size, latent_dim]

    # Compute log q(z) (marginalized over batch)
    log_qz = torch.logsumexp(log_qz_cond_x.sum(dim=1, keepdim=True), dim=0) - torch.log(torch.tensor(batch_size, dtype=torch.float, device=z_sample.device))

    # Compute log q(z_j) for each dimension (marginal distribution)
    log_qz_product = torch.sum(torch.logsumexp(log_qz_cond_x, dim=0) - torch.log(torch.tensor(batch_size, dtype=torch.float, device=z_sample.device)))

    # Compute Total Correlation loss
    tc_loss = (log_qz - log_qz_product).mean()
    
    return tc_loss


def beta_tcvae_lossT(decoded, x, mu, log_var, z_sample, n_nodes, alpha, beta, gamma, delta, atlas):
    """
    Loss for β-TCVAE: reconstruction loss + Laplacian smoothing + KL divergence + Total Correlation.

    Parameters:
        decoded: Reconstructed output.
        x: Original input.
        mu: Latent mean.
        log_var: Latent log variance.
        z_sample: Sampled latent variable.
        n_nodes: Number of valid nodes in mesh.
        alpha: Weight for Laplacian loss.
        beta: Weight for KL divergence.
        gamma: Weight for normal consistency loss.
        delta: Weight for TC loss (to enforce latent independence).
        atlas: Atlas mesh file (used for Laplacian smoothing).

    Returns:
        loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss
    """

    recon_loss = nn.MSELoss()(decoded.view(x.size(0), -1), x.view(x.size(0), -1))
    
    batch_size, C, T, H, W = x.shape
    
    atlas = pv.read(atlas)  # Load atlas mesh
    
    faces = torch.tensor(atlas.regular_faces, dtype=torch.long).to(decoded.device)  # Extract mesh faces

    # Reshape inputs to match mesh-based processing
    decoded = decoded.permute(0, 2, 3, 4, 1).reshape(batch_size * T, H * W, 3)[:, :n_nodes, :]
    x = x.permute(0, 2, 3, 4, 1).reshape(batch_size * T, H * W, 3)[:, :n_nodes, :]

    # Compute Laplacian and normal losses
    meshes = Meshes(verts=decoded, faces=faces.unsqueeze(0).expand(decoded.shape[0], -1, -1))
    lap_loss = lap_dis(meshes)
    norm_loss = normal_dis(meshes)

    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

    # Total Correlation loss
    tc_loss = compute_total_correlation(z_sample, mu, log_var)

    # Final loss function
    loss = recon_loss + alpha * lap_loss + beta * kl_loss + gamma * norm_loss + delta * tc_loss

    return loss, recon_loss, kl_loss, lap_loss, norm_loss, tc_loss


########################################################
def beta_ae_lossT(decoded, x, n_nodes, alpha, beta,  gamma, atlas, args):
    """
    Loss for b-VAE: reconstruction loss + Laplacian smoothing + KL divergence.

    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    recon_loss = nn.MSELoss()(decoded.view(x.size(0), -1), x.view(x.size(0), -1))
    
    batch_size, C, T, H, W = x.shape
    
    # print(x.shape)
    # print(decoded.shape)

    # atlas = pv.read('/ukb/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec/avg/LVmyo_mesh_avg_decim_0.95_0.vtk')
    atlas = pv.read(atlas)
 
    # Correctly extract faces from PyVista
    ''' 
    PolyData.regular_faces in PyVista automatically removes 
    the leading count (e.g., 3 for triangles) and returns the face connectivity in a clean format.
    '''
    faces = torch.tensor(atlas.regular_faces, dtype=torch.long).to(device)


    # faces = np.array(atlas.faces.reshape(-1, 4)[:, 1:])  # Convert faces format
    # faces = torch.tensor(faces, dtype=torch.long).to(device)  # Move to GPU

    # **Apply the same reshaping method used in reverse_preprocessing()**
    decoded = decoded.permute(0, 2, 3, 4, 1)  # [B, C, T, H, W] → [B, T, H, W, C]
    x = x.permute(0, 2, 3, 4, 1)  # [B, C, T, H, W] → [B, T, H, W, C]


    ## Temporal Coherence: Smoothness Loss
    # temporal_smoothness_loss = 0
    # for t in range(T - 1):
    #     temporal_smoothness_loss += F.mse_loss(decoded[:,:, t + 1, :, :], decoded[:,:, t, :, :])
    # temporal_smoothness_loss /= (T - 1)
    
    ## Temporal Coherence: Smoothness Loss  (vectorized)
    temporal_diffs = decoded[:, :, 1:, :, :] - decoded[:, :, :-1, :, :]
    temporal_smoothness_loss = torch.mean(temporal_diffs ** 2)



    # decoded = decoded.view(batch_size * T, H * W, 3)  # [BT, H*W, C]
    # x = x.view(batch_size * T, H * W, 3)  # [BT, H*W, C]


    decoded = decoded.reshape(batch_size * T, H * W, 3)  # [BT, H*W, 3]
    x = x.reshape(batch_size * T, H * W, 3)  # [BT, H*W, 3]


    decoded = decoded[:, :n_nodes, :]  # Remove padding
    x = x[:, :n_nodes, :]  # Remove padding
    
    
    


    # **Now, compute Chamfer Distance on the correctly reshaped data**
    # recon_loss, _ = chamfer_dis(decoded, x)
    # chamfer_loss, _ = chamfer_dis(decoded, x)

    # a = pv.PolyData(x[0, :, :].cpu().numpy(), atlas.faces)
    # a.save ('/skalaie/motion_code/DR_motion/a.vtk')
    
    # b = pv.PolyData(decoded[0, :, :].detach().cpu().numpy(), atlas.faces)
    # b.save ('/skalaie/motion_code/DR_motion/b.vtk')
    

    # print("Faces using reshape method:\n", atlas.faces.reshape(-1, 4)[:, 1:][:5])
    # print("Faces using regular_faces:\n", atlas.regular_faces[:5])
    # print("Extracted regular_faces tensor:\n", faces[:5])
    # print("Original atlas faces:\n", atlas.faces[:12])  # First few faces from PyVista
    # print('F=', faces.unsqueeze(0).expand(batch_size * T, -1, -1))
    
    # Compute Laplacian loss
    meshes = Meshes(verts=decoded, faces=faces.unsqueeze(0).expand(decoded.shape[0], -1, -1))
    lap_loss = lap_dis(meshes)
    
    
    norm_loss = normal_dis(meshes)  ## prevent disjointed smoothing effects.
    
    # sparse_loss=   torch.mean(torch.abs(encoded))
    # KL divergence loss

    # Total loss
 
    # loss = recon_loss + alpha * lap_loss + beta * kl_loss 
    loss = recon_loss + alpha * lap_loss + beta * temporal_smoothness_loss + gamma * norm_loss 
    # loss = recon_loss + 0.1 * chamfer_loss + alpha * lap_loss + beta * kl_loss + gamma * norm_loss
    return loss, recon_loss,  lap_loss, norm_loss, temporal_smoothness_loss


def beta_vae_lossT(decoded, x, mu, log_var, n_nodes, alpha, beta, gamma, delta, atlas, args):
    """
    Loss for b-VAE: reconstruction loss + Laplacian smoothing + KL divergence + Normal loss + Temporal smoothing.

    """
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    recon_loss = nn.MSELoss()(decoded.view(x.size(0), -1), x.view(x.size(0), -1))
    
    batch_size, C, T, H, W = x.shape
    
    # print(x.shape)
    # print(decoded.shape)

    # atlas = pv.read('/ukb/UKBB_40616_deepmesh/cine_meshes_deepmesh_dec/avg/LVmyo_mesh_avg_decim_0.95_0.vtk')
    atlas = pv.read(atlas)
 
    # Correctly extract faces from PyVista
    ''' 
    PolyData.regular_faces in PyVista automatically removes 
    the leading count (e.g., 3 for triangles) and returns the face connectivity in a clean format.
    '''
    faces = torch.tensor(atlas.regular_faces, dtype=torch.long).to(device)


    # faces = np.array(atlas.faces.reshape(-1, 4)[:, 1:])  # Convert faces format
    # faces = torch.tensor(faces, dtype=torch.long).to(device)  # Move to GPU

    # **Apply the same reshaping method used in reverse_preprocessing()**
    decoded = decoded.permute(0, 2, 3, 4, 1)  # [B, C, T, H, W] → [B, T, H, W, C]
    x = x.permute(0, 2, 3, 4, 1)  # [B, C, T, H, W] → [B, T, H, W, C]

    # decoded = decoded.view(batch_size * T, H * W, 3)  # [BT, H*W, C]
    # x = x.view(batch_size * T, H * W, 3)  # [BT, H*W, C]


    ## Temporal Coherence: Smoothness Loss  (vectorized)
    temporal_diffs = decoded[:, :, 1:, :, :] - decoded[:, :, :-1, :, :]
    temporal_smoothness_loss = torch.mean(temporal_diffs ** 2)


    decoded = decoded.reshape(batch_size * T, H * W, 3)  # [BT, H*W, 3]
    x = x.reshape(batch_size * T, H * W, 3)  # [BT, H*W, 3]


    decoded = decoded[:, :n_nodes, :]  # Remove padding
    x = x[:, :n_nodes, :]  # Remove padding
    
    # Compute Laplacian loss
    meshes = Meshes(verts=decoded, faces=faces.unsqueeze(0).expand(decoded.shape[0], -1, -1))
    lap_loss = lap_dis(meshes)
    
    
    norm_loss = normal_dis(meshes)  ## prevent disjointed smoothing effects.
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

    

    # Total loss
    loss = recon_loss + alpha * lap_loss + beta * kl_loss + gamma * norm_loss + delta * temporal_smoothness_loss

    return loss, recon_loss, kl_loss, lap_loss, norm_loss, temporal_smoothness_loss

####### Rec+KL ####################################################
def beta_vae_loss(decoded, x, mu, log_var, beta):
    """
    Loss for b-VAE: reconstruction loss + b * KL divergence.
    """
    # Reconstruction loss (MSE or binary cross-entropy)
    recon_loss = nn.MSELoss()(decoded.view(x.size(0), -1), x.view(x.size(0), -1))
    
    # # Reshape decoded and input (x) to match Chamfer Distance format
    # batch_size, C, T, H, W = x.shape
    # decoded_points = decoded.view(batch_size, -1, C)  # Flatten to [batch_size, num_points, 3]
    # input_points = x.view(batch_size, -1, C)  # Flatten to [batch_size, num_points, 3]
    
    # decoded_reshaped = decoded.view(batch_size * T, H*W, 3)
    # x_reshaped = x.view(batch_size * T, H*W, 3)
    # recon_loss = chamfer_dis(decoded_reshaped, x_reshaped)[0]

    # Reconstruction loss (Chamfer Distance)
    # recon_loss, _ = chamfer_dis(decoded_points, input_points)
    
    
    # Temporal Coherence: Smoothness Loss
    # temporal_smoothness_loss = 0
    # for t in range(T - 1):
    #     temporal_smoothness_loss += F.mse_loss(decoded[:,:, t + 1, :, :], decoded[:,:, t, :, :])
    # temporal_smoothness_loss /= (T - 1)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
    # beta = 3e-5
    # beta = 1e-5
    # beta_t = 0.0001
    # Combine losses
    
    return recon_loss +  beta * kl_loss, recon_loss, kl_loss
    # return recon_loss + beta_t * temporal_smoothness_loss+ beta * kl_loss, recon_loss, kl_loss
    
    
def save_loss_plot(train_losses, val_losses, output_dir, epoch, name):
    """
    Save the training and validation loss plot as an image.

    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        output_dir: Directory to save the plot image.
        epoch: The current epoch or 'final' to indicate the final plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o", color="b")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="x", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Save plot as an image
    plot_filename = os.path.join(output_dir, f"{name}_plot_epoch_{epoch}.png")
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to free memory
    print(f"Loss plot saved: {plot_filename}")


# def adaptive_beta(epoch, start_beta, max_beta, warmup_epochs, mode="linear"):
#     """Increase beta gradually over epochs to balance reconstruction & disentanglement."""
    
#     ## A fter warmup_epochs, β reaches  max_beta and stays constant.
#     if epoch < warmup_epochs:
#         scale = epoch / warmup_epochs
#     else:
#         scale = 1.0
    
#     if mode == "linear":
#         return start_beta + (max_beta - start_beta) * scale
#     elif mode == "exponential":
#         return start_beta * (max_beta / start_beta) ** scale
#     return start_beta


def adaptive_beta(epoch, start_beta, max_beta, warmup_epochs, grow_epochs=10, mode="linear"):
    """
    Gradually increase beta after a warmup period focused only on reconstruction.
    - warmup_epochs: how many epochs to keep beta = 0
    - grow_epochs: how many epochs to increase beta from start_beta to max_beta
    """
    if epoch < warmup_epochs:
        return 0.0  # Pure reconstruction
    scale = (epoch - warmup_epochs) / grow_epochs
    scale = min(scale, 1.0)  # Cap at 1.0
    
    if mode == "linear":
        return start_beta + (max_beta - start_beta) * scale
    elif mode == "exponential":
        return start_beta * (max_beta / start_beta) ** scale
    
    return start_beta
    
def compute_latent_correlation(output_dir, fig_name, latents):
    """Computes correlation matrix of latents to check for disentanglement."""

    latents = np.concatenate(latents, axis=0)  # Stack all latents
    corr_matrix = np.corrcoef(latents.T)  # Compute correlation

    # Plot heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    # sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=0, vmax=1)

    plt.title("Latent Space Correlation Matrix")
    # plt.show()
    plt.savefig(f"{output_dir}/{fig_name}.png")

    # return corr_matrix




# def plot_training_curves_ae(alpha_values, beta_values, train_losses, val_losses, train_recon_losses, val_recon_losses, train_lap_losses, val_lap_losses, output_dir):
#     """
#     Plots the evolution of alpha, beta, and losses over training epochs.
#     """
#     epochs = range(1, len(alpha_values) + 1)

#     # Plot alpha and beta values
#     plt.figure(figsize=(12, 5))
#     plt.plot(epochs, alpha_values, label="Alpha", color="blue")
#     plt.plot(epochs, beta_values, label="Beta", color="orange")
#     plt.plot(epochs, val_losses, label="Validation Loss", color="green")
#     plt.xlabel("Epochs")
#     plt.ylabel("Value")
#     plt.title("Alpha and Beta Values Over Epochs")
#     plt.legend()
#     plt.grid()
#     plt.savefig(f"{output_dir}/alpha_beta_evolution.png")
#     plt.close()

#     # Plot losses
#     plt.figure(figsize=(12, 5))
#     plt.plot(epochs, train_losses, label="Train Loss", color="blue")
#     plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Total Loss Over Epochs")
#     plt.legend()
#     plt.grid()
#     plt.savefig(f"{output_dir}/total_loss_evolution.png")
#     plt.close()

#     # Plot reconstruction loss
#     plt.figure(figsize=(12, 5))
#     plt.plot(epochs, train_recon_losses, label="Train Recon Loss", color="blue")
#     plt.plot(epochs, val_recon_losses, label="Validation Recon Loss", color="orange")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Reconstruction Loss Over Epochs")
#     plt.legend()
#     plt.grid()
#     plt.savefig(f"{output_dir}/recon_loss_evolution.png")
#     plt.close()


#     # Plot Laplacian loss
#     plt.figure(figsize=(12, 5))
#     plt.plot(epochs, train_lap_losses, label="Train Laplacian Loss", color="blue")
#     plt.plot(epochs, val_lap_losses, label="Validation Laplacian Loss", color="orange")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Laplacian Loss Over Epochs")
#     plt.legend()
#     plt.grid()
#     plt.savefig(f"{output_dir}/lap_loss_evolution.png")
#     plt.close()

def plot_training_curves_ae(alpha_values, beta_values, train_losses, val_losses, train_recon_losses, val_recon_losses, train_lap_losses, val_lap_losses, train_norm_losses, val_norm_losses, output_dir):
    """
    Plots the evolution of alpha, beta, and losses over training epochs.
    """
    epochs = range(1, len(alpha_values) + 1)

    # Plot alpha and beta values
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, alpha_values, label="Alpha", color="blue")
    plt.plot(epochs, beta_values, label="Beta", color="orange")
    plt.plot(epochs, val_losses, label="Validation Loss", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Alpha and Beta Values Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/alpha_beta_evolution.png")
    plt.close()

    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Total Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/total_loss_evolution.png")
    plt.close()

    # Plot reconstruction loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_recon_losses, label="Train Recon Loss", color="blue")
    plt.plot(epochs, val_recon_losses, label="Validation Recon Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/recon_loss_evolution.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_norm_losses, label="Train Normal Loss", color="blue")
    plt.plot(epochs, val_norm_losses, label="Validation Normal Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Normal Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/norm_loss_evolution.png")
    plt.close()
    
    # Plot Laplacian loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_lap_losses, label="Train Laplacian Loss", color="blue")
    plt.plot(epochs, val_lap_losses, label="Validation Laplacian Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Laplacian Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/lap_loss_evolution.png")
    plt.close()


def plot_training_curves_ae_all( train_losses, val_losses, train_recon_losses, val_recon_losses, train_lap_losses, val_lap_losses, train_norm_losses, val_norm_losses, train_t_losses, val_t_losses, output_dir):
    """
    Plots the evolution of alpha, beta, and losses over training epochs.
    """
    epochs = range(1, len(train_losses) + 1)


    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Total Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/total_loss_evolution.png")
    plt.close()

    # Plot reconstruction loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_recon_losses, label="Train Recon Loss", color="blue")
    plt.plot(epochs, val_recon_losses, label="Validation Recon Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/recon_loss_evolution.png")
    plt.close()

    # Plot KL loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_t_losses, label="Train Temporal Loss", color="blue")
    plt.plot(epochs, val_t_losses, label="Validation Temporal Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Temporal Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/temporal_loss_evolution.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_norm_losses, label="Train Normal Loss", color="blue")
    plt.plot(epochs, val_norm_losses, label="Validation Normal Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Normal Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/norm_loss_evolution.png")
    plt.close()
    
    # Plot Laplacian loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_lap_losses, label="Train Laplacian Loss", color="blue")
    plt.plot(epochs, val_lap_losses, label="Validation Laplacian Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Laplacian Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/lap_loss_evolution.png")
    plt.close()


def plot_training_curves_all( beta_values, train_losses, val_losses, train_recon_losses, val_recon_losses, train_kl_losses, val_kl_losses, train_lap_losses,val_lap_losses, train_norm_losses, val_norm_losses, train_t_losses, val_t_losses, output_dir):
    """
    Plots the evolution of alpha, beta, and losses over training epochs.
    """
    epochs = range(1, len(beta_values) + 1)

    # Plot alpha and beta values
    plt.figure(figsize=(12, 5))
    # plt.plot(epochs, alpha_values, label="Alpha", color="blue")
    plt.plot(epochs, beta_values, label="Beta", color="orange")
    plt.plot(epochs,val_kl_losses, label="Validation KL Loss", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    # plt.title("Alpha and Beta Values Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/alpha_beta_evolution.png")
    plt.close()

    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Total Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/total_loss_evolution.png")
    plt.close()

    # Plot reconstruction loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_recon_losses, label="Train Recon Loss", color="blue")
    plt.plot(epochs, val_recon_losses, label="Validation Recon Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/recon_loss_evolution.png")
    plt.close()

    # Plot KL loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_kl_losses, label="Train KL Loss", color="blue")
    plt.plot(epochs, val_kl_losses, label="Validation KL Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("KL Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/kl_loss_evolution.png")
    plt.close()

    # Plot Laplacian loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_lap_losses, label="Train Laplacian Loss", color="blue")
    plt.plot(epochs, val_lap_losses, label="Validation Laplacian Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Laplacian Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/lap_loss_evolution.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_norm_losses, label="Train Normal Loss", color="blue")
    plt.plot(epochs, val_norm_losses, label="Validation Normal Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Normal Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/norm_loss_evolution.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_t_losses, label="Train Temporal Loss", color="blue")
    plt.plot(epochs, val_t_losses, label="Validation Temporal Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Temporal Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/temporal_loss_evolution.png")
    plt.close()
# def save_checkpoint(output_dir, modelName, model, optimizer, loss_train, loss_val, epoch):
    
#     weight_prefix = os.path.join(output_dir, modelName)

#     #    torch.save(model.state_dict(), weight_prefix)
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss_train': loss_train,
#         'loss_val': loss_val,
#     }, weight_prefix)
    
    
def save_checkpoint(output_dir, modelName, model, optimizer, loss_train, loss_val, epoch):
    # Create the checkpoints directory if it doesn't exist
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Build full path to the checkpoint file
    checkpoint_path = os.path.join(checkpoints_dir, modelName)

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': loss_train,
        'loss_val': loss_val,
    }, checkpoint_path)
