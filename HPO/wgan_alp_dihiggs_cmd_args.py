
# coding: utf-8

# In[2]:

import subprocess
from dihiggs_dataset import DiHiggsSignalMCDataset
subprocess.call(["python", "dihiggs_dataset.py"])

import argparse
import os
import numpy as np
import math
import sys
from collections import OrderedDict

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F   # NOTE: I don't think this is used
import torch.autograd as autograd
import torch
from torch import Tensor

from IPython import display
import logging
from matplotlib import pyplot as plt

import process_output_HPO

#
print('CUDA is available: ' + str(torch.cuda.is_available()))


# In[6]:

#os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--n_epochs", type=int, help="number of epochs of training")
parser.add_argument("--batchSize", type=int, help="size of the batches")
parser.add_argument("--learningRate", type=float, help="adam: learning rate")
parser.add_argument("--beta1", type=float, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latentSpaceSize", type=int, help="dimensionality of the latent space")
#parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--lrDecayRate", type=float, help="Learning rate decay rate")
parser.add_argument("--width", type=int, help="Width of network")
parser.add_argument("--depth", type=int, help="Depth of network")
parser.add_argument("--activationFunction", type=str, help="Activation function used")
parser.add_argument("--alpha", type=float, help="Alpha value for LeakyRelu (if used)")
parser.add_argument("--batch_norm", type=bool, help="True or false value for using batch normalization")
parser.add_argument("--configDir", type=str, help="Directory to save data to")

opt = parser.parse_args()
print(opt)


class opt_old():   # Class used for optimizers in the future. Defines all variables and stuff needed.
    n_epochs = 20000   # an epoch is the number of times it works through the entire training set
    batch_size = 1000   # the training set is broken up into batches, 
                        # and the average loss is used from a given batch for back propagation
    lr =  0.0002 # 0.001   # learning rate (how much to change based on error)
    b1 = 0     # 0.9 # Used for Adam. Exponential decay rate for the first moment. 
    b2 = 0.9   # 0.999 # Used for Adam. Exponential decay rate for the second moment estimates (gradient squared)
    #NOTE: The default epsilon for torch.optim.adam is 1e-8, so I will just leave it that way
    
    #n_cpu = 2   # not used rn
    latent_dim = 100 #size of noise input to generator (latent space) 
    #img_size = 28
    # channels = 1   # Only used for img_shape right below, and img_shape isn't needed
    n_critic = 5   # The generator is trained after this many critic steps
    #   clip_value = 0.01   # No other usages rn. 
    sample_interval = 400   # Determines when a to save the image(s?) generated
    
    Xi = 10;   # multiplier for recursively finding r_adversarial
    
    # Loss weight for alp penalty
    lambda_alp = 100

# img_shape = (opt.channels, opt.img_size, opt.img_size)   # Not used rn

cuda = True if torch.cuda.is_available() else False

sample_interval = 400
n_critic = 5

try:
    os.makedirs(opt.configDir)
except OSError:
    logging.warning("Output folders already exist. May overwrite some output files.")

class Generator(nn.Module):
    """
    Create hidden layers. Apply normalization. Apply leaky relu. 
    """
    def __init__(self):
        super(Generator, self).__init__()   

        def block(in_feat, out_feat, alpha=0.2, normalize=True):   # This function creates the hidden layers
            layers = [nn.Linear(in_feat, out_feat)]   # layer is a hidden layer. Takes input
                                                      # (batch_size,in_feat) and give an output (batch_size,out_feat)
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))   # adds normalization to what Layers does to input and comes out in
                                                               # size (batch_size,out_feat). I think this does bn1d(linear(input))
            layers.append(nn.LeakyReLU(alpha, inplace=True))   # inplace means just modify input, don't allocate more memory
            return layers



        """
        stores layers and functions applied to layers
        """   
        if os.path.exists(opt.configDir + "/generatorModel.pt"): 
            self.model = torch.load(opt.configDir + "/generatorModel.pt")
        else:
            layers = OrderedDict()
            for i in range(opt.depth):
                if i == 0:
                    layers['layer_' + str(i)] = block(opt.latent_dim, opt.width, alpha=opt.alpha, normalize=opt.batch_norm)
                elif i == opt.depth - 1:
                    layers['layer_' + str(i)] = nn.Linear(opt.width, 25)
                    layers['tanh'] = nn.Tanh()
                else:
                    layers['layer_' + str(i)] = block(opt.width, opt.width, alpha=opt.alpha, normalize=opt.batch_norm)
                
            self.model = nn.Sequential(layers)

    def forward(self, z):
        """
        applies layers to input to get img
        """
        img = self.model(z)   # applies model (layers and functions on layers) to z
        #img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    """
    Discriminator/critic layers
    """
    def __init__(self):
        super(Discriminator, self).__init__()   # Just uses the module constructor with name Discriminator 

        def block(in_feat, out_feat, alpha=0.2, normalize=True):   # This function creates the hidden layers
            layers = [nn.Linear(in_feat, out_feat)]   # layer is a hidden layer. Takes input
                                                      # (batch_size,in_feat) and give an output (batch_size,out_feat)
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))   # adds normalization to what Layers does to input and comes out in
                                                               # size (batch_size,out_feat). I think this does bn1d(linear(input))
            layers.append(nn.LeakyReLU(alpha, inplace=True))   # inplace means just modify input, don't allocate more memory
            return layers

        if os.path.exists(opt.configDir + "/discriminatorModel.pt"): 
            self.model = torch.load(opt.configDir + "/discriminatorModel.pt")
        else:
            layers = OrderedDict()
            for i in range(opt.depth):
                if i == 0:
                    layers['layer_' + str(i)] = block(25, opt.width, alpha=opt.alpha, normalize=False)
                elif i == opt.depth - 1:
                    layers['layer_' + str(i)] = nn.Linear(opt.width, 1)
                else:
                    layers['layer_' + str(i)] = block(opt.width, opt.width, alpha=opt.alpha, normalize=False)
                

            self.model = nn.Sequential(layers)

    def forward(self, img):
        """
        applies model to image and gives validity score
        """
        img_flat = img.view(img.shape[0], -1)   # TODO: Figure out what this does 
        validity = self.model(img_flat)   # calculates validity score
        #print("forward validity from discriminator: " + str((np.max(np.abs(validity.detach().numpy())))))
        return validity


# ******* OUT OF CLASSES NOW ************

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    
    
# Configure data loader - CHANGE
os.makedirs("./data/", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
   DiHiggsSignalMCDataset('./DiHiggs Data', generator_level = False),
   batch_size=opt.batch_size,
   shuffle=True,
)
print('done')


# In[5]:

# In[4]:

def compute_ALP(D, real_samples, fake_samples):   # TODO: Find out why these are .data
    """
    Calculates the gradient penalty loss for WGAN GP
    D input will be discrimantor function
    real_samples and fake_samples are from reality and generator. Both are sent in via memory location of buffer
    
    """
    
    # Random weight term for interpolation between real and fake samples (how much of each)
    alpha = Tensor(np.random.random((real_samples.size(0),1)))   # This is a tensor designating which to use where
    #print(alpha)
  #  print(alpha.shape)
    # Get random interpolation between real and fake samples
   # print(real_samples.shape)
    
    # Gets some of real and some of fake samples for gradient penalty calculation
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # .requires grad is something attached to all tensors and can be used to speed up (by making false I think)
    # It is automatically false, but if you need gradient then set to be true
    # TODO: Understand how this statement works
    
    
    ################## CALCULATE R ADVERSARIAL ###############################################
    # start with random unit vector r0
    r0 = np.random.rand(interpolates.shape[0], interpolates.shape[1])
    r0 = Tensor(r0/r0.max(axis = 0)).requires_grad_(True)
    #print(r[0])
    
    #  add this initial r to our random data points
    interpol_y0 = (interpolates + opt.Xi * r0).requires_grad_(True)   #.requires_grad_(True)
    # run the discriminator on both of these
    d_interpolates = D(interpolates)   # Run discriminator on interpolates to get validity scores
    d_interpol_y0 = D(interpol_y0)   # do the same for the adjusted interpolates to find r adversarial

    
    # find gradient(d(f(x) - f(x+r)))
    difference = (d_interpolates - d_interpol_y0).requires_grad_(True)  #.requires_grad_(True)
    #print("d interpolates: " + str(d_interpolates.shape) + " " + str(d_interpolates.type))
    #print("difference: " + str(difference.shape) + " " + str(difference.type))
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False) 
    gradient_r0 = autograd.grad(
        outputs=difference,
        inputs=r0,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # finally, find r_adversarial!
    epsilon_r = np.random.uniform(0.1,10)
    r_adv = epsilon_r * gradient_r0/np.linalg.norm(gradient_r0.cpu().detach().numpy())
    #print(np.max(np.linalg.norm(r_adv.cpu().detach().numpy())))
###########################################################################################################

######### Now find the loss ###########################
    
    interpol_adversarial = (interpolates + r_adv).requires_grad_(True)
    d_interpol_adv = D(interpol_adversarial)
    abs_difference = np.abs((d_interpolates - d_interpol_adv).cpu().detach().numpy())/     (np.linalg.norm(r_adv.cpu().detach().numpy())) - 1
    squared = np.square(np.maximum(abs_difference,np.zeros(100)))
    #print("Max of alp before mean: " + str(np.max(np.abs(squared))))
    
    alp_penalty = squared.mean()
   # print("ALP final: " + str(alp_penalty))
    
    return alp_penalty

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))   
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Create learning rate decay schedulers
my_lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_G, gamma=opt.lrDecayRate)
my_lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_D, gamma=opt.lrDecayRate)

# In[ ]:

batches_done = 0   # Counter for batches
for epoch in range(opt.n_epochs):   # Loop through all epochs
    for i, x in enumerate(dataloader): # x is in dataloader (a batch I think). i
                                       # is the index of x (number of times critic is trained this epoch)

        # Configure input
        real_imgs = Variable(x.type(Tensor))   # Variable is a wrapper for the Tensor x was just made into

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()   # Make gradients zero so they don't accumulate

        # Sample noise (latent space) to make generator input
        z = Variable(Tensor(np.random.normal(0, 1, (x.shape[0], opt.latent_dim))))   # Once again Variable wraps the Tensor
#         print(type(x))
#         print(x.shape)
#         print(x[0].shape())
#         print(z.shape)

        # Generate a batch of images from the latent space sampled
        fake_imgs = generator(z)

        #print(fake_imgs[0])

        # Calculate validity score for real images
        real_validity = discriminator(real_imgs)

        # Calculate validity score for fake images
        fake_validity = discriminator(fake_imgs)

        # Calculate gradient penalty
        alp = compute_ALP(discriminator, real_imgs.data, fake_imgs.data)
        # TODO: figure out why .data is used

        # Calculate loss for critic (Adversarial loss)
        d_loss = torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_alp * alp

        d_loss.backward()   # Do back propagation 
        optimizer_D.step()   # Update parameters based on gradients for individuals

        optimizer_G.zero_grad()   # Resets gradients for generator to be zero to avoid accumulation

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()


            # ----------------------------
            # Save stuff when time is right
            # ----------------------------
            if batches_done % 10 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

            if batches_done % sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += n_critic

    # Call learning rate decays every epoch
    my_lr_scheduler_D.step()
    my_lr_scheduler_G.step()

    if epoch % 10 == 0:
        z = Variable(Tensor(np.random.normal(0, 1, (300000, opt.latent_dim))))
        fake_data = generator(z)
        np.save('./gen_data_alp/{num_batches}.npy'.format(num_batches=batches_done), fake_data.cpu().detach().numpy())

torch.save(generator.model, opt.configDir + "/generatorModel.pt")
torch.save(discriminator.model, opt.configDir + "/discriminatorModel.pt")

process_output_and_metric(opt.configDir)

