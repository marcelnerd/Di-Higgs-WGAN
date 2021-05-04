#!/usr/bin/env python
import math
import random
import process_output_HPO   # check that I did this correctly
import sys
import os
import logging
import Hyperparameter_Optimization

def run_then_return_metric(config, r_i, run_num):
    os.system("cd " + config)
    os.system("echo " + str(r_i) + " >> run_num.txt")
    os.system("sbatch -A scholar --nodes=1 --gpus=1 -t 0:05:00 ./training_sub_file.sub >> ../output.txt")
    os.system("rm run_num.txt")
    os.system("cd ../../../")

def top_k_performers(T, metrics, k):
    metrics, T = heapSort(metrics, T)

    return T[:k], metrics[:k]

parser = argparse.ArgumentParser()

parser.add_argument("--i", type=int, help="number of epochs of training")
parser.add_argument("--s", type=float, help="outer loop number... needed for configs")
parser.add_argument("--n", type=int, help="number of configurations")
parser.add_argument("--r", type=float, help="per bracket base runtime")
parser.add_argument("--eta", type=float, help="1/eta is fraction we keep")
parser.add_argument("--configFile", type=str, help="Name of file with configurations listed")


opt = parser.parse_args()

i = opt.i
s = opt.s
configFileName = opt.configFile
# if i > 0:
#     T = []
#     metric_file = open("metric_" + str(s) + ".txt", "r")
#     file_stuff = metric_file.readline()
#     count = 0
#     while file_stuff:
#         T[count] = file_stuff.split(" ")[1].rstrip("\n")
#         count += 1
#         file_stuff = metric_file.readline()
    
if i > 0: # Change this so it reads from the metric file in the right folder maybe?
    metrics = []
    T = []
    metric_file = open("metric_" + str(s) + "_" + str(i) + ".txt","r")
    file_stuff = metric_file.readline()
    count = 0
    while file_stuff:
        metrics[count] = file_stuff.split(" ")[0]
        T[count] = file_stuff.split(" ")[1].rstrip("\n")
        count += 1
        file_stuff = metric_file.readline()
    metric_file.close()

    T, metrics = top_k_performers(T, metrics, math.floor(n_i / eta))
    T = [t.split("_i_", 1)[0] + "_i_" + str(i) for t in T] # Update the directory with the new i
else:
    T = []
    config_file = open(configFileName, "r")
    file_stuff = config_file.readline()
    count = 0
    while file_stuff:
        T[count] = file_stuff.rstrip("\n")
        count += 1
        file_stuff = config_file.readline()

    config_file.close()
    T = [t + "_i_" + i for t in T]

#ith_configs = [t for t in T if 'iteration_' + str(i) in t]
# make the directories
# for directory in T:
#     os.system("mkdir " + directory)
#     try:
#         os.makedirs(os.path.join("./", directory))
#     except OSError:
#         logging.warning("Output folders already exist. May overwrite some output files.")

print("i: " + str(i)) 
n_i = int(math.floor(n * eta**(-1 * i)))
r_i = int(r * eta**i)
for t in T:
    run_then_return_metric(t, r_i, i)

# create job ID list
jobIDs = ""
job_ID_file = open("output.txt","r")
jobID = job_ID_file.readline()
count = 0

while jobID:
    jobIDs += jobID.split(" ")[3].rstrip("\n")
    if (count < 4 - 1):
        jobIDs += ","
        count += 1
    jobID = job_ID_file.readline()

#create dependancy job
os.system("sbatch -A scholar --nodes=1 --gpus=1 -t 0:05:00 --dependency=afterok:" + jobIDs + " ./recursive.sub")