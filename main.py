# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 18:31:28 2021

@author: drago
"""
CUDA_LAUNCH_BLOCKING="1"

import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model3s #imports model.py (linearVAE)
import models #imports the CVAE models
import pandas as pd
import math
from sklearn.metrics import r2_score
import matplotlib.ticker as mtick

# ROC plot stuff
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils

from tqdm import tqdm
from numpy import zeros, newaxis
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

matplotlib.style.use('ggplot')

#Forces Double to match float64
torch.set_default_dtype(torch.float64)

# %%  Learning Parameters
max_epoch = 250

# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=max_epoch, type=int, 
                    help='number of epochs to train the VAE for')
args = vars(parser.parse_args())

# leanring parameters
epochs = args['epochs']
batch_size = 100
# lr = 0.0001
lr = 5e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')

# # %% My Data
# df = pd.read_excel('VAE Generation 1.xlsx') # All Data
df = pd.read_excel('Reduced Data2.xlsx') #Uniform Distribution
# df = pd.read_excel('Top.xlsx')  #Top 500 PF Values
df_perm = df

# Separates data 
#==================================
Run = df['Run ']
ID = df['ID ']
df = df.drop(['Run '],axis=1)               #Remove .stat file number


# Global Scale
#=====================
# dft = df.iloc[:,1:7]
# datascale =100000
# dft=dft/datascale

# dftt = pd.concat([df.iloc[:,0:1],dft,df.iloc[:,7:18]], axis=1)

# Individual Scaling
#=====================
df1 = df.iloc[:,1]
df1m = df1.max(0)
df1 = df1/df1m

df2 = df.iloc[:,2]
df2m = df2.max(0)
df2 = df2/df2m

df3 = df.iloc[:,3]
df3m =df3.max(0)
df3 = df3/df3m

df4 = df.iloc[:,4]
df4m =df4.max(0)
df4 = df4/df4m

df5 = df.iloc[:,5]
df5m = df5.max(0)
df5 = df5/df5m

df6 = df.iloc[:,6]
df6m = df6.max(0)
df6 = df6/df6m

# # dt =df.iloc[:,7:18]

dftt = pd.concat([df.iloc[:,0:1],df1,df2,df3,df4,df5,df6,df.iloc[:,7:18]], axis=1)

df = dftt
#
# =============================================================================
# MOST EXCEPTIONAL SETUP
# =============================================================================
              
df_main = df
df_main.sort_values(by='Packing_Fraction ',ascending=False)

#                 Main Test Train Split
# =============================================================================
cutoff = 499        #number of exceptional values
split = 0.1        #percentage used for testing

exceptional = df_main.iloc[0:cutoff, :]
normal = df_main.iloc[cutoff+1 :, :]

df_extra1 = exceptional.sample(frac=split,replace=False)
df_extra2 = exceptional[~exceptional.isin(df_extra1)].dropna()

df_norm1 = normal.sample(frac=split,replace=False)
df_norm2 = normal[~normal.isin(df_norm1)].dropna()

df_test = pd.concat([df_extra1, df_norm1])                  #TESTING DATA
df_train_intermediate = pd.concat([df_extra2, df_norm2])

#                 Training Data Split
# =============================================================================
df_train_intermediate.sort_values(by='Packing_Fraction ',ascending=False)

cutoff2 = int(cutoff*(1-split))         #Number of exceptional passed into training data
excep_train = df_train_intermediate.iloc[0:cutoff2, :]  #remainder of exceptional 
norm_train = df_train_intermediate.iloc[cutoff2+1 :, :] #remainder of normal

split2 = 0.5    #splits the data evenly
df_extra_val = excep_train.sample(frac=split2,replace=False)
df_extra_train = excep_train[~excep_train.isin(df_extra_val)].dropna()

df_norm_val = norm_train.sample(frac=split2,replace=False)
df_norm_train = norm_train[~norm_train.isin(df_norm_val)].dropna()


df_validate = pd.concat([df_extra_val, df_norm_val])        #VALIDATION DATA

#==============================================================================
df_training = pd.concat([df_extra_train, df_norm_train])    #TRAINING DATA


df_validate_y = df_validate.iloc[:,-1]                         #Validate Packing Fraction
df_validate = df_validate.drop(['Packing_Fraction '],axis=1)   #Validate Inputs
df_validate = df_validate.drop(['ID '],axis=1)

df_test = df_test.drop(['ID '],axis=1)
df_test_st = df_test
df_test_y = df_test.iloc[:,-1]                         #Validate Packing Fraction
df_test = df_test.drop(['Packing_Fraction '],axis=1)   #Validate Inputs

df_train_intermediate = df_train_intermediate.drop(['ID '],axis=1)               #Remove .stat file number
df_train_y = df_train_intermediate.iloc[:,-1]                         #Validate Packing Fraction
df_train = df_train_intermediate.drop(['Packing_Fraction '],axis=1)   #Validate Inputs

# %% Data Loader

#Convert to Double
train_temp = df_train_intermediate.astype(np.float64)
df_train_intermediate = df_train_intermediate.astype(np.float64)
df_test_st = df_test_st.astype(np.float64)

#Convert to Tensor
train_data = torch.from_numpy(df_train_intermediate.values)
val_data = torch.from_numpy(df_test_st.values)

    
# training and validation data loaders
train_loader = DataLoader(
    train_data,
    batch_size=len(df_train_intermediate),
    shuffle=True
)
val_loader = DataLoader(
    val_data,
    batch_size=len(df_test_st),
    shuffle=False
)

# %% Initialize the Model, Optimizer, and Loss Function
model = model3s.LinearVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')
criterion2 = nn.MSELoss(reduction='sum')
# criterion = nn.BCEWithLogitsLoss(reduction='sum')

# beta = 0
# beta = 1
# beta = 4
# beta = 7
beta = 10
# beta = 15
# beta = 20
# beta = 25
# beta = 50

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    Final = KLD+BCE*beta
    
    # print('\n BCE', BCE.shape)
    # print('\n KLD', KLD.shape)
    return BCE, KLD, Final      #Total Loss 

# %%  Training Function
# actually runs and gathers the information
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def fit(model, dataloader):
    model.train()
    running_loss_bce = 0.0
    running_loss_kld = 0.0
    running_loss = 0.0
    
    #loops thru data in intervals of batch sizes
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/(dataloader.batch_size/batch_size))):
        # print('i',i)
        data = dataloader.dataset
        data = data.to(device)
        data = data.view(data.size(0), -1)
        
        #Normalization of the data [0,1]
        data = torch.nn.functional.normalize(data)
       
        #zero the parameter gradients
        optimizer.zero_grad()
        
        #forward + backward + optimize
        reconstruction, mu, logvar = model(data)
 
        #Choice of BCE or MSE
        bce_loss = criterion2(reconstruction, data)   #MSE
        # bce_loss = criterion(reconstruction, data)  #BCE
        [loss_bce, loss_kld, loss] = final_loss(bce_loss, mu, logvar)
        
        running_loss_bce += loss_bce.item()
        # loss_bce.backward(retain_graph=True)
        
        running_loss_kld += loss_kld.item()
        # loss_kld.backward(retain_graph=True)
        
        running_loss+= loss
        loss.backward()
        optimizer.step()
        
        # print('Reconstruction', reconstruction.shape)
        
    train_loss_bce = running_loss_bce/len(dataloader.dataset)
    train_loss_kld = running_loss_kld/len(dataloader.dataset)
    return train_loss_bce, train_loss_kld, reconstruction, data



# %% Validation Function

def validate(model, dataloader):
    model.eval()
    running_loss_bce = 0.0
    running_loss_kld = 0.0
    running_loss = 0.0
    # with torch.no_grad():
    for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
        data = dataloader.dataset
        data = data.to(device)
        data = data.view(data.size(0), -1)
        
        #Normalization of the data [0,1]
        data = torch.nn.functional.normalize(data)
       
        #zero the parameter gradients
        optimizer.zero_grad()
        
        #forward + backward + optimize
        reconstruction, mu, logvar = model(data)
 
        #Choice of BCE or MSE
        bce_loss = criterion2(reconstruction, data)   #MSE
        # bce_loss = criterion(reconstruction, data)  #BCE
        [loss_bce, loss_kld, loss] = final_loss(bce_loss, mu, logvar)
        
        running_loss_bce += loss_bce.item()
        # loss_bce.backward(retain_graph=True)
        
        running_loss_kld += loss_kld.item()
        # loss_kld.backward(retain_graph=True)
        
        running_loss+= loss
        loss.backward()
        optimizer.step()
        
        # print('Reconstruction', reconstruction.shape)
        
    train_loss_bce = running_loss_bce/len(dataloader.dataset)
    train_loss_kld = running_loss_kld/len(dataloader.dataset)
    return train_loss_bce, train_loss_kld, reconstruction, data

# %%  Main
# Trains and Validates data

train_kld = []     #KLD Loss
val_kld = []
train_bce = []      #BCE Loss
val_bce = []
train_loss_tot = []      #total Loss
val_loss_tot = []
train_recon = []      #Reconstructed Data
val_recon = []
train_Data = []      #Reconstructed Data
val_Data = []
rtrain = []      #Reconstructed Data
rtest = []
for epoch in range(epochs):
    print(f"          Epoch {epoch+1} of {epochs}")
    
    # if epoch==50:
    #     lr=.1

    [train_epoch_bce, train_epoch_kld, train_recon_epoch, traindata_epoch,] = fit(model, train_loader)
    [val_epoch_bce, val_epoch_kld, val_recon_epoch, valdata_epoch,] = validate(model, val_loader)
    
    #train loss and data
    train_kld.append(train_epoch_kld)    
    train_bce.append(train_epoch_bce)
    train_recon.append(train_recon_epoch.cpu().detach().numpy())
    train_Data.append(traindata_epoch.cpu().detach().numpy())
    
    #validation loss and data
    val_kld.append(val_epoch_kld)
    val_bce.append(val_epoch_bce)
    val_recon.append(val_recon_epoch.cpu().detach().numpy())
    val_Data.append(valdata_epoch.cpu().detach().numpy())
    
    #total loss
    val_loss_tot.append(val_epoch_bce+val_epoch_kld)
    train_loss_tot.append(train_epoch_bce+train_epoch_kld)
    
    Rtest=r2_score(train_recon[epoch][:,9],df_train_y)
    Rtrain=r2_score(val_recon[epoch][:,9],df_test_y)
    
    rtrain.append(Rtrain)
    rtest.append(Rtest)
    

# %% Plotting - Epoch Loss

fig = plt.figure(1)
ax1 = fig.add_subplot(1,1,1)

vl, = plt.plot(val_loss_tot,'b')
tl, = plt.plot(train_loss_tot,'r')

bs = 'Batch Size  ' + str(batch_size)
lnr = '   Learning Rate   ' + str(lr)
eph = '   Max Epoch   ' + str(max_epoch)
var = bs+lnr+eph


ax1.grid(False)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')

fs = 10
# plt.title('VAE Loss - Particle Packing 6-Sigma\n{}'.format(var),
#           fontsize = fs, weight = 'bold')
# plt.xlabel('Epoch', fontsize = fs, weight = 'bold')
# plt.ylabel('Epoch Loss', fontsize = fs, weight = 'bold')

ax1.set_ylabel('Epoch Loss', color='black', fontsize=fs, weight='bold')
ax1.set_xlabel('Epoch', color='black', fontsize=fs, weight='bold')

plt.legend([tl, vl,],
           ['Train Loss Total', 'Val Loss Total'],
           shadow=True, fancybox=True, fontsize = fs, loc='best', facecolor='white')

ax1.set_facecolor('white')

ax1.set_box_aspect(1)


# ax1.rcParams["axes.edgecolor"] = "black"
# ax1.rcParams["axes.linewidth"] = 1
    
# plt.axis('square')

# plt.ylim(0,10000)

# plt.savefig(var+'.png')
# plt.close()


# %% TSNE Stuff
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

c = df.columns
c = c[1:11] 

final_data = pd.DataFrame(val_recon[max_epoch -1])
final_data.set_axis(c, axis=1, inplace=True)

x_test = final_data.iloc[:,0:9]
y_test = final_data.iloc[:,-1]


# We want to get TSNE embedding with 2 dimensions
n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(x_test)
tsne_result.shape
# (1000, 2)
# Two dimensions for each of our images
 
# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y_test})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
plt.title('Final Epoch Render: ' r'$\beta$ =' +str(beta))
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# -----------------------------------------------------
#                     Original Data
# -----------------------------------------------------

# We want to get TSNE embedding with 2 dimensions
n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(df_train_intermediate.iloc[:,0:9])
tsne_result.shape
# (1000, 2)
# Two dimensions for each of our images
 
# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': df_train_y})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
plt.title('Original Data')
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


# %%
# from torch.autograd import Variable
# import tensorflow as tf
# import torch

# # def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None):
    
# xten =  x_test.astype(np.float64)
# xten = torch.tensor(xten.values)
# qz_samples = xten
# qz_params = xten
# q_dist = xten
# n_samples = 10000
# weights = None
# """Computes the term:
#     E_{p(x)} E_{q(z|x)} [-log q(z)]
# and
#     E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
# where q(z) = 1/N sum_n=1^N q(z|x_n).
# Assumes samples are from q(z|x) for *all* x in the dataset.
# Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).
# Computes numerically stable NLL:
#     - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)
# Inputs:
# -------
#     qz_samples (K, N) Variable
#     qz_params  (N, K, nparams) Variable
#     weights (N) Variable
# """
# # Only take a sample subset of the samples
# if weights is None:
#     qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples]))
#     # qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
# else:
#     sample_inds = torch.multinomial(weights, n_samples, replacement=True)
#     qz_samples = qz_samples.index_select(1, sample_inds)

# K, S = qz_samples.size()
# N, _, nparams = qz_params.size()
# assert(nparams == q_dist.nparams)
# assert(K == qz_params.size(1))

# if weights is None:
#     weights = -math.log(N)
# else:
#     weights = torch.log(weights.view(N, 1, 1) / weights.sum())

# entropies = torch.zeros(K).cuda()

# pbar = tqdm(total=S)
# k = 0
# while k < S:
#     batch_size = min(10, S - k)
#     logqz_i = q_dist.log_density(
#         qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
#         qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
#     k += batch_size

#     # computes - log q(z_i) summed over minibatch
#     entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
#     pbar.update(batch_size)
# pbar.close()

# entropies /= S





# %% ROC stuff

# final_data = pd.DataFrame(val_recon[max_epoch -1])
# x_test = final_data.iloc[:,0:9]
# y_test = final_data.iloc[:,-1]


# # ROC stuff

# #Convert to binary
# lab = preprocessing.LabelEncoder()
# y_traint = lab.fit_transform(df_train_y)
# y_testt = lab.fit_transform(y_test)

# label_binarizer = LabelBinarizer().fit(y_traint)
# y_onehot_test = label_binarizer.transform(y_testt)


# classifier = LogisticRegression()
# y_score = classifier.fit(df_train, y_traint).predict_proba(x_test)

# RocCurveDisplay.from_predictions(
#     y_onehot_test[:,1], y_score[:,1])

# plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
# plt.legend()
# plt.show()

# %%Save last prediction 
# ans = pd.concat([df2,y_predicted], axis=1, sort=False)
k = max_epoch -1
ans =val_recon[k]
ans = pd.DataFrame(ans)
c = df.columns
c = c[1:11] 

ans.set_axis(c, axis=1, inplace=True)
ans0 = ans.iloc[:,0]*df1m
ans1 = ans.iloc[:,1]*df2m
ans2 = ans.iloc[:,2]*df3m
ans3 = ans.iloc[:,3]*df4m
ans4 = ans.iloc[:,4]*df5m
ans5 = ans.iloc[:,5]*df6m

# ans0 = ans.iloc[:,0]
# ans1 = ans.iloc[:,1]
# ans2 = ans.iloc[:,2]
# ans3 = ans.iloc[:,3]
# ans4 = ans.iloc[:,4]
# ans5 = ans.iloc[:,5]

export = pd.concat([ans0,ans1, ans2, ans3, ans4, ans5, ans.iloc[:,6:10]], axis=1)
export.to_excel("VAE Predictions.xlsx")

