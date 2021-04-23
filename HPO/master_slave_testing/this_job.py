#!/usr/bin/env python
import argparse
import numpy as np
from process_output_HPO import process_output_and_metric
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, help="path to directory")
opt = parser.parse_args()
current_dir = opt.dir
processed_dir = current_dir.split('/')[-2].split('_')
args = {'learning_rate': processed_dir[2], 'lr_decay_rate': processed_dir[6], 'width': processed_dir[8], 'depth': processed_dir[10], 'activation_function': processed_dir[13], 'alpha': processed_dir[15], 'latent_space_size': processed_dir[19], 'batch_size': processed_dir[22], 'batch_norm': processed_dir[25], 'beta1':processed_dir[27], 'beta2':processed_dir[29], 'iteration':processed_dir[31]}

num_run_file = open("num_run.txt","r") 
num_epochs = num_run_file.readline()

os.system("python wgan_alp_dihiggs_cmd_args.py --n_epochs " + str(num_epochs) \
    + " --learningRate " + args['learning_rate'] \
    + " --lrDecayRate " + args['lr_decay_rate'] \
    + " --batchSize " + args['batch_size'] \
    + " --bawidthtchSize " + args['width'] \
    + " --depth " + args['depth'] \
    + " --activationFunction " + args['activation_function'] \
    + " --alpha " + args['alpha'] \
    + " --latentSpaceSize " + args['latent_space_size'] \
    + " --batch_norm " + args['batch_norm'] \
    + " --beta1 " + args['beta1'] \
    + " --beta2 " + args['beta2'])


#print(str(i ** 2) + " on the {j}th iteration".format(j = i))
#np.save("./" + '{j}.npy'.format(j = i), 2*i)