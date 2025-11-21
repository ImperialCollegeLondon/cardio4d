
import pyvista as pv
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, r2plus1d_18
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



################################################################################################################################
class Cardio4DVAE(nn.Module):
    def __init__(self, encoded_dim, input_shape):
        super(Cardio4DVAE, self).__init__()
  

        # Pretrained 3D ResNet as encoder
        self.encoder = r3d_18(weights="DEFAULT")
        print('r3d_18')
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 2 * encoded_dim)  # Output both mu and log(var)

        # Calculate flattened input size
        self.input_shape = input_shape
        self.input_size = torch.prod(torch.tensor(input_shape)).item()
        # print('input_shape' ,input_shape)  ## (3, 25, 34, 35)
        # print('input_shape' , self.input_size) ## 89250
        # Lightweight decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_size),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample latent variables z ~ N(mu, sigma^2).
        """
        std = torch.exp(0.5 * log_var)  # Convert log variance to standard deviation
        eps = torch.randn_like(std)  # Random normal noise
        return mu + eps * std

    def forward(self, x):
        # Encoder: Outputs mu and log(var)
        encoded = self.encoder(x)
        mu, log_var = encoded.chunk(2, dim=-1)  # Split into mean and log variance

        # Reparameterization trick
        z = self.reparameterize(mu, log_var)

        # Decoder: Reconstruct input
        decoded = self.decoder(z)
        
        # print('Decoder size', decoded.shape)  ## torch.Size([batch, 89250])
        batch_size = x.size(0)
        decoded = decoded.view(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        # print('model input:',x.shape)  ## torch.Size([10, 3, 25, 34, 35])
        # print('model output:',decoded.shape) ## torch.Size([10, 3, 25, 34, 35])                                                                    
        return mu, log_var, z, decoded
    
    
    
class I3DAutoencoder(nn.Module):
    def __init__(self, encoded_dim, input_shape):  ##(3, 25, 34, 35)
        super(I3DAutoencoder, self).__init__()
        # Pretrained 3D ResNet as encoder
        self.encoder = r3d_18(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, encoded_dim)

        
        self.input_shape = input_shape  # Save for reshaping
        # Calculate the flattened input size
        self.input_size = torch.prod(torch.tensor(input_shape)).item()
        # Lightweight decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_size),  # Match flattened input size
            nn.Sigmoid()  # Ensure values are between 0 and 1
        )

    def forward(self, x):
        # Encoder: Reduce dimensionality
        encoded = self.encoder(x)

        # Decoder: Reconstruct input
        decoded = self.decoder(encoded)
        
        # Reshape decoded back to [batch_size, C, T, H, W]  
        batch_size = x.size(0)
        decoded = decoded.view(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])  ##[batch_size, 3, 50, 37, 38]  
        
        return encoded, decoded
    
################################################################################################################################
    
# # simple version
# class BetaVAE(nn.Module):
#     def __init__(self, encoded_dim, input_shape):
#         super(BetaVAE, self).__init__()
  

#         # Pretrained 3D ResNet as encoder
#         # self.encoder = r3d_18(weights="DEFAULT")
#         self.encoder = r2plus1d_18(weights="DEFAULT")
#         print('r2plus1d_18')

#         self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 2 * encoded_dim)  # Output both mu and log(var)
#         self.dropout = nn.Dropout(0.3)  # Dropout after encoding: dropout_rate=0.3

#         # Calculate flattened input size
#         self.input_shape = input_shape
#         self.input_size = torch.prod(torch.tensor(input_shape)).item()
#         # print('input_shape' ,input_shape)  ## (3, 25, 34, 35)
#         # print('input_shape' , self.input_size) ## 89250
#         # Lightweight decoder for reconstruction
#         self.decoder = nn.Sequential(
#             nn.Linear(encoded_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             # nn.Linear(512, 256),
#             # nn.ReLU(),
#             nn.Linear(512, self.input_size),
#             nn.Sigmoid()
#         )

#     def reparameterize(self, mu, log_var):
#         """
#         Reparameterization trick to sample latent variables z ~ N(mu, sigma^2).
#         """
#         std = torch.exp(0.5 * log_var)  # Convert log variance to standard deviation
#         eps = torch.randn_like(std)  # Random normal noise
#         return mu + eps * std

#     def forward(self, x):
#         # Encoder: Outputs mu and log(var)
#         encoded = self.encoder(x)
#         # encoded = self.dropout(encoded)  # Regularize latent space

#         mu, log_var = encoded.chunk(2, dim=-1)  # Split into mean and log variance

#         # Reparameterization trick
#         z = self.reparameterize(mu, log_var)

#         # Decoder: Reconstruct input
#         decoded = self.decoder(z)
        
#         # print('Decoder size', decoded.shape)  ## torch.Size([batch, 89250])
#         batch_size = x.size(0)
#         decoded = decoded.view(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
#         # print('model input:',x.shape)  ## torch.Size([10, 3, 25, 34, 35])
#         # print('model output:',decoded.shape) ## torch.Size([10, 3, 25, 34, 35])                                                                    
#         return mu, log_var, z, decoded




##########################################################################################################################    


class AE(nn.Module):
    def __init__(self, encoded_dim, input_shape):   
        super(AE, self).__init__()
        # Pretrained 3D ResNet as encoder
        # self.encoder = r3d_18(pretrained=True)
        # self.encoder = r3d_18(weights="DEFAULT")
        self.encoder = r2plus1d_18(weights="DEFAULT")
        print('r2plus1d_18')
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, encoded_dim)

        self.dropout = nn.Dropout(0.3)  # Dropout after encoding: dropout_rate=0.3
        
        
        self.input_shape = input_shape  # Save for reshaping
        # Calculate the flattened input size
        self.input_size = torch.prod(torch.tensor(input_shape)).item()
        # print(self.input_size)
        # # Calculate flattened input size
        # self.input_shape = input_shape  # Save for reshaping
        # self.input_size = int(np.prod(input_shape))

        # Lightweight decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_size),  # Match flattened input size
            nn.Sigmoid()  # Ensure values are between 0 and 1
        )

    def forward(self, x):
        # print(x.shape)
        # Encoder: Reduce dimensionality
        encoded = self.encoder(x)
        encoded = self.dropout(encoded)  # Regularize latent space

        # Decoder: Reconstruct input
        decoded = self.decoder(encoded)
        
        # Reshape decoded back to [batch_size, C, T, H, W]  
        batch_size = x.size(0)
        decoded = decoded.view(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])  ##[batch_size, 3, 50, 37, 38]  
        
        # print(decoded.shape)
        return encoded, decoded   
##########################################################################################################################    
    
    
    
    # class BetaVAE(nn.Module):
#     def __init__(self, encoded_dim, input_shape):
#         super(BetaVAE, self).__init__()
  

#         # Pretrained 3D ResNet as encoder
#         self.encoder = r3d_18(weights="DEFAULT")
#         self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 2 * encoded_dim)  # Output both mu and log(var)

#         # Calculate flattened input size
#         self.input_shape = input_shape
#         self.input_size = torch.prod(torch.tensor(input_shape)).item()
#         # print('input_shape' ,input_shape)  ## (3, 25, 34, 35)
#         # print('input_shape' , self.input_size) ## 89250
#         # Lightweight decoder for reconstruction
        
        
#         # self.decoder = nn.Sequential(
#         #     nn.Linear(encoded_dim, 1024),
#         #     nn.ReLU(),
#         #     nn.Linear(1024, self.input_size),
#         #     nn.Sigmoid()
#         # )
#         # Structured Intermediate Shape
#         self.intermediate_shape = (3, 6, 12, 12)  # Example intermediate shape
#         self.intermediate_size = torch.prod(torch.tensor(self.intermediate_shape)).item()  # Flattened size

#         # Improved Fully Connected Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(encoded_dim, 512),
#             nn.ReLU(),

#             nn.Linear(512, 2048),
#             nn.ReLU(),


#             nn.Linear(2048, self.intermediate_size),   # Reshape to (3, 6, 12, 12)
#             nn.ReLU(),

#             nn.Linear(self.intermediate_size, self.input_size),   #  (3, 6, 12, 12) to (3, 25, 34, 35)
#             nn.Sigmoid()  # Normalize output between [0,1]
#         )

#     def reparameterize(self, mu, log_var):
#         """
#         Reparameterization trick to sample latent variables z ~ N(mu, sigma^2).
#         """
#         std = torch.exp(0.5 * log_var)  # Convert log variance to standard deviation
#         eps = torch.randn_like(std)  # Random normal noise
#         return mu + eps * std

#     def forward(self, x):
#         # Encoder: Outputs mu and log(var)
#         encoded = self.encoder(x)
#         mu, log_var = encoded.chunk(2, dim=-1)  # Split into mean and log variance

#         # Reparameterization trick
#         z = self.reparameterize(mu, log_var)

#         # Decoder: Reconstruct input
#         decoded = self.decoder(z)
        
#         # print('Decoder size', decoded.shape)  ## torch.Size([batch, 89250])
#         batch_size = x.size(0)
#         decoded = decoded.view(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
#         # print('model input:',x.shape)  ## torch.Size([10, 3, 25, 34, 35])
#         # print('model output:',decoded.shape) ## torch.Size([10, 3, 25, 34, 35])                                                                    
#         return mu, log_var, z, decoded
    
################################################################################################################################

import torch.nn.functional as F
import torch.nn as nn

#####  Shallow Decoder  ####################################

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoder, self).__init__()

        # Fully Connected layer maps latent space to small spatial representation
        self.fc = nn.Linear(latent_dim, 512 * 3 * 3 * 3)  # Expanding to small 3D feature map
        
        
        C, T, H, W = output_shape
        self.init_T = T // 16
        self.init_H = H // 16
        self.init_W = W // 16

        self.fc = nn.Linear(latent_dim, 512 * self.init_T * self.init_H * self.init_W)

        # Deconvolution layers to upsample correctly
        self.deconv1 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)  # (B, 256, 6, 6, 6)
        self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)  # (B, 128, 12, 12, 12)
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)   # (B, 64, 24, 24, 24)
        # self.deconv4 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))   # torch.Size([10, 32, 47, 48, 48])
        # self.deconv5 = nn.ConvTranspose3d(32, 3, kernel_size=1)  # Final output: (B, 3, 25, 34, 35)

        # Normalization layers to improve training stability
        self.batch_norm1 = nn.BatchNorm3d(256)
        self.batch_norm2 = nn.BatchNorm3d(128)
        self.batch_norm3 = nn.BatchNorm3d(64)
        # self.batch_norm4 = nn.BatchNorm3d(32)

        # # Fully Connected layer for final shape correction
        # self.fc_final = nn.Linear(32 * 47 * 48 * 48 ,  3 * 25 * 34 * 35) 
        # **Adaptive Convolution for Exact Output Shape**
        self.final_conv = nn.Conv3d(64, 3, kernel_size=3, stride=1, padding=1)  # (B, 3, 48, 48, 48)
        # self.upsample = nn.AdaptiveAvgPool3d((25, 34, 35))  # Enforce exact shape
        self.upsample = nn.AdaptiveAvgPool3d((50, 34, 35))  # Enforce exact shape



    def forward(self, z):
        x = self.fc(z)  # Shape: [B, 512 * 3 * 3 * 3]
        x = x.view(-1, 512, 3, 3, 3)  # Reshape to small feature map

        # Apply transpose convolutions with normalization
        x = F.relu(self.batch_norm1(self.deconv1(x)))  # (B, 256, 6, 6, 6)
        # print(x.shape)
        x = F.relu(self.batch_norm2(self.deconv2(x)))  # (B, 128, 12, 12, 12)
        # print(x.shape)
        x = F.relu(self.batch_norm3(self.deconv3(x)))  # (B, 64, 24, 24, 24)
        # print(x.shape)
        x = F.relu(self.batch_norm4(self.deconv4(x)))  # (B, 32, 25, 34, 35)
        # print(x.shape)
        # print(self.deconv5(x).shape)
        # x = torch.sigmoid(self.deconv5(x))  # Normalize output

        # return x  # Shape: (B, 3, 25, 34, 35)
        
        # Final Convolution & Adaptive Resizing
        x = self.final_conv(x)  # (B, 3, 48, 48, 48)
        x = self.upsample(x)  # (B, 3, 25, 34, 35)

        return torch.sigmoid(x)  # Normalize output
    
        # # Flatten and apply FC layer to get correct shape
        # x = x.view(x.shape[0], -1)  # Flatten to (B, 32*47*48*48)
        # x = self.fc_final(x)  # Map to (B, 3*25*34*35)
        # x = x.view(-1, 3, 25, 34, 35)  # Reshape to final shape
        # return torch.sigmoid(x)  # Normalize output


############ mirror decoder (deep) ########################

# class ConvDecoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(ConvDecoder, self).__init__()

#         # Fully Connected layer maps latent space to small spatial representation
#         self.fc = nn.Linear(latent_dim, 512 * 3 * 3 * 3)  # Expanding to small 3D feature map


#         # Deconvolution layers to upsample correctly
#         self.deconv1 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)  # (B, 256, 6, 6, 6)
#         self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)  # (B, 128, 12, 12, 12)
#         self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)   # (B, 64, 24, 24, 24)
#         self.deconv4 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))   # torch.Size([10, 32, 47, 48, 48])
#         # self.deconv5 = nn.ConvTranspose3d(32, 3, kernel_size=1)  # Final output: (B, 3, 25, 34, 35)

#         # Normalization layers to improve training stability
#         self.batch_norm1 = nn.BatchNorm3d(256)
#         self.batch_norm2 = nn.BatchNorm3d(128)
#         self.batch_norm3 = nn.BatchNorm3d(64)
#         self.batch_norm4 = nn.BatchNorm3d(32)

#         # # Fully Connected layer for final shape correction
#         # self.fc_final = nn.Linear(32 * 47 * 48 * 48 ,  3 * 25 * 34 * 35) 
#         # **Adaptive Convolution for Exact Output Shape**
#         self.final_conv = nn.Conv3d(32, 3, kernel_size=3, stride=1, padding=1)  # (B, 3, 48, 48, 48)
#         # self.upsample = nn.AdaptiveAvgPool3d((25, 34, 35))  # Enforce exact shape
#         self.upsample = nn.AdaptiveAvgPool3d((50, 34, 35))  # Enforce exact shape



#     def forward(self, z):
#         x = self.fc(z)  # Shape: [B, 512 * 3 * 3 * 3]
#         x = x.view(-1, 512, 3, 3, 3)  # Reshape to small feature map

#         # Apply transpose convolutions with normalization
#         x = F.relu(self.batch_norm1(self.deconv1(x)))  # (B, 256, 6, 6, 6)
#         # print(x.shape)
#         x = F.relu(self.batch_norm2(self.deconv2(x)))  # (B, 128, 12, 12, 12)
#         # print(x.shape)
#         x = F.relu(self.batch_norm3(self.deconv3(x)))  # (B, 64, 24, 24, 24)
#         # print(x.shape)
#         x = F.relu(self.batch_norm4(self.deconv4(x)))  # (B, 32, 25, 34, 35)
#         # print(x.shape)
#         # print(self.deconv5(x).shape)
#         # x = torch.sigmoid(self.deconv5(x))  # Normalize output

#         # return x  # Shape: (B, 3, 25, 34, 35)
        
#         # Final Convolution & Adaptive Resizing
#         x = self.final_conv(x)  # (B, 3, 48, 48, 48)
#         x = self.upsample(x)  # (B, 3, 25, 34, 35)

#         return torch.sigmoid(x)  # Normalize output
    
#         # # Flatten and apply FC layer to get correct shape
#         # x = x.view(x.shape[0], -1)  # Flatten to (B, 32*47*48*48)
#         # x = self.fc_final(x)  # Map to (B, 3*25*34*35)
#         # x = x.view(-1, 3, 25, 34, 35)  # Reshape to final shape
#         # return torch.sigmoid(x)  # Normalize output



#####  MirrorDecoder  ####################################
## javab bad nist accurate nist vali shape LV darad


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(Decoder, self).__init__()
        C, T, H, W = output_shape
        print(T)
        self.init_T = T // 16
        self.init_H = H // 16
        self.init_W = W // 16

        self.fc = nn.Linear(latent_dim, 512 * self.init_T * self.init_H * self.init_W)

        self.decoder_blocks = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),  # T/8, H/8, W/8
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # T/4, H/4, W/4
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # T/2, H/2, W/2
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),  # T, H, W
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, C, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool3d((T, H, W)),  # Enforce exact shape
            # nn.Upsample(size=(T, H, W), mode='trilinear', align_corners=False),  # Upsample instead of Pool

            # nn.AdaptiveAvgPool3d((50, 34, 35))  # Enforce exact shape
            # nn.Sigmoid()  # Final output normalized
        )

    def forward(self, z):
        x = self.fc(z)
        # print(x.shape)
        x = x.view(-1, 512, self.init_T, self.init_H, self.init_W)
        # print(x.shape)
        x = self.decoder_blocks(x)
        # print(x.shape)
        return x


##############################


###############################################################################################################
 ## upsampling Temporal+spatial 
 ## only spatial upsampling after first layer.
 #########################################################################
# '''
# torch.Size([10, 512, 25, 2, 2])
# torch.Size([10, 256, 50, 4, 4])
# torch.Size([10, 128, 50, 8, 8])
# torch.Size([10, 64, 50, 16, 16])
# torch.Size([10, 3, 50, 32, 32]) 
# torch.Size([10, 3, 50, 34, 35])
 
# '''
# class Decoder(nn.Module):
#     def __init__(self, latent_dim, output_shape):
#         super(Decoder, self).__init__()
#         C, T, H, W = output_shape
        
#         # Calculate flattened input/output size
#         self.output_shape = output_shape
#         self.output_size = torch.prod(torch.tensor(output_shape)).item()

#         self.init_T = T // 2  # Only 1 temporal upsample needed
#         self.init_H = H // 16
#         self.init_W = W // 16

#         self.fc = nn.Linear(latent_dim, 512 * self.init_T * self.init_H * self.init_W)
        
#         self.up4 = nn.Sequential(
#             nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),  # Fine for (2,2,2) stride
#             nn.BatchNorm3d(256),
#             nn.ReLU()
#         )

#         # up3: Spatial-only upsample (reverse Conv3D(3,3,3), stride=(1,2,2))
#         self.up3 = nn.Sequential(
#             nn.ConvTranspose3d(256, 128, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),  # More accurate than (1,4,4)
#             nn.BatchNorm3d(128),
#             nn.ReLU()
#         )

#         # up2: Same as up3
#         self.up2 = nn.Sequential(
#             nn.ConvTranspose3d(128, 64, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
#             nn.BatchNorm3d(64),
#             nn.ReLU()
#         )

#         # Final: Spatial only
#         self.up1 = nn.Sequential(
#             nn.ConvTranspose3d(64, C, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1))
#         )
#         # # Upsample temporally + spatially (first layer mirrors first encoder downsample)
#         # self.up4 = nn.Sequential(
#         #     nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample T, H, W
#         #     nn.BatchNorm3d(256),
#         #     nn.ReLU()
#         # )

#         # # Only spatial upsampling from here (stride=(1,2,2))
#         # self.up3 = nn.Sequential(
#         #     nn.ConvTranspose3d(256, 128, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
#         #     nn.BatchNorm3d(128),
#         #     nn.ReLU()
#         # )

#         # self.up2 = nn.Sequential(
#         #     nn.ConvTranspose3d(128, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
#         #     nn.BatchNorm3d(64),
#         #     nn.ReLU()
#         # )

#         # self.up1 = nn.Sequential(
#         #     nn.ConvTranspose3d(64, C, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
#         #     nn.BatchNorm3d(64),
#         #     nn.ReLU()
#         # )

#         # self.final_conv = nn.Sequential(
#         #     nn.AdaptiveAvgPool3d((T, H, W)),  ##(3, 25, 34, 35)
#         #     # nn.Linear(64, self.output_size),   #  (3, 6, 12, 12) to (3, 25, 34, 35)
#         #     nn.Sigmoid()
#         # )
        
        
#         # Structured Intermediate Shape
#         self.intermediate_shape = (3, 50, 32, 32)  # Example intermediate shape
#         self.intermediate_size = torch.prod(torch.tensor(self.intermediate_shape)).item()  # Flattened size
        
#         self.final_conv = nn.Sequential(
#             nn.Linear(self.intermediate_size, self.output_size),  ##(3, 50, 34, 35)
#             nn.Sigmoid()
#         )
#         # Final non-linear activation (Sigmoid for [0,1] output)
#         # self.final_activation = nn.Sigmoid()
        
#         # self.final_upsample = nn.Sequential(
#         #     nn.Upsample(size=(T, H, W), mode='trilinear', align_corners=False),
#         #     nn.Conv3d(C, C, kernel_size=3, padding=1),
#         #     nn.Sigmoid()
#         # )

        

#     def forward(self, z):
#         x = self.fc(z)
#         x = x.view(-1, 512, self.init_T, self.init_H, self.init_W)
#         # print(x.shape)
#         x = self.up4(x)  # Temporal + spatial upsample
#         # print(x.shape)
#         x = self.up3(x)  # Spatial only
#         # print(x.shape)
#         x = self.up2(x)
#         # print(x.shape)
#         x = self.up1(x)
#         # print(x.shape)
#         x = self.final_conv(x)
        
#         # x = F.pad(x, (0, 3, 0, 2))  # (W_left, W_right, H_top, H_bottom)
        
#         # # print(x.shape)
#         # x = self.final_activation(x)
        
#         batch_size = x.size(0)
#         decoded = decoded.view(batch_size,self.output_shape[0], self.output_shape[1], self.output_shape[2], self.output_shape[3])# x = self.final_upsample(x)
        
#         # print(x.shape)
#         return x
#########################################################################################################################



# class BetaVAE(nn.Module):
#     def __init__(self, encoded_dim, input_shape):
#         super(BetaVAE, self).__init__()
        
#         """
#         Encodes the input by passing through the convolutional network
#         and outputs the latent variables.

#         Params:
#             input (Tensor): Input tensor [B x C x H x W]

#         Returns:
#             mu (Tensor) and log_var (Tensor) of latent variables
#         """

#         # Pretrained 3D ResNet as encoder
#         self.encoder = r3d_18(weights="DEFAULT")
#         self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 2 * encoded_dim)  # Output both mu and log(var)

#         # Save input shape
#         self.input_shape = input_shape
#         self.input_size = torch.prod(torch.tensor(input_shape)).item()

#         # New convolutional decoder
#         # self.decoder = ConvDecoder(encoded_dim)
#         self.decoder = Decoder(encoded_dim, self.input_shape)

#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         # print("input:", x.shape)
#         encoded = self.encoder(x)
#         # print("encoded:", encoded.shape)
#         mu, log_var = encoded.chunk(2, dim=-1)
#         z = self.reparameterize(mu, log_var)

#         # Reconstruct using new decoder
#         decoded = self.decoder(z)
#         # # print('Decoder size', decoded.shape)  ## torch.Size([batch, 89250])
#         # batch_size = x.size(0)
#         # decoded = decoded.view(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
#         # print('model input:',x.shape)  ## torch.Size([10, 3, 25, 34, 35])
#         # print("decoded:", decoded.shape)
#         return mu, log_var, z, decoded
#  #########################################################################################################################
