# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:52:18 2021

@author: drago
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#Forces Double to match float64
torch.set_default_dtype(torch.float64)

# %%  LinearVAE() Module
# ==============================================
features = 3
# define a simple linear VAE
class LinearVAE(nn.Module):
    #Object to create the layering
    def __init__(self):   
        super(LinearVAE, self).__init__()
         
        # encoder
        self.enc1 = nn.Linear(in_features=10, out_features=6)
        self.enc2 = nn.Linear(in_features=6, out_features=features*2)
                
        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=6)
        self.dec2 = nn.Linear(in_features=6, out_features=10)
        
        # self.norm = nn.LayerNorm(3)
                
    #object to sample the latent space like the input space
    def reparameterize(self, mu, log_var): 
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        
        # sample = (sample - mu) / std #normalization
        
        # print('Std Max', std.max())
        # print('Epsilon Max', log_var.max(),'\n')
        return sample
        
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)
               
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        
        # get the latent vector through reparameterization
        std = torch.exp(0.5*log_var.max()) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        # print('STD:  ', std.max())
        
        z = sample

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        
        #normalizes the mass fraction sum to equal 1
        ## ======================================
        a, mf, vf = torch.split(reconstruction, [6,3,1], dim=1)
        # mf = self.norm(mf)
        mf=nn.functional.normalize(mf, p=1, dim=1)
        reconstruction = torch.cat([a,mf,vf], dim=1)

        return reconstruction, mu, log_var
    # ====================================================