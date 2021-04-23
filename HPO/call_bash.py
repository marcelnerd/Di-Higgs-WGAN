#!/usr/bin/env python
import subprocess
import os
import numpy as np
import time

from test_funcs import get_hyperparameter_configurations

for s in range(0,3):
    jobIDs = ""

    
    for i in range(0,4):
        
        #os.system("sbatch -A partner" + " --nodes=1 --gpus=1 -t 0:05:00 ./submission.sub >> output.txt")
        os.system("sbatch -A scholar" + " --nodes=1 --gpus=1 -t 0:05:00 -i " + str(i) + " ./submission.sub >> output.txt")
        os.system("echo s = " + str(s) + ", and i = " + str(i) + ".")
        #time.sleep(0.01)
        #os.system("cat output.txt")
    #os.system("cat output.txt")
    file = open("output.txt","r") 
    line = file.readline()
    count = 0
    while line:
        jobIDs += line.split(" ")[3].rstrip("\n")
        if (count < 4 - 1):
            jobIDs += ","
            count += 1
        line = file.readline()
    print(jobIDs)
    file.close()
    #os.system("sbatch -A partner --nodes=1 --gpus=1 -t 0:05:00 --dependency=afterok:" + jobIDs + " ./empty.sub")
    os.system("sbatch -A scholar --nodes=1 --gpus=1 -t 0:05:00 --dependency=afterok:" + jobIDs + " ./empty.sub")
    os.system("rm output.txt")
    #load saved stuff and sort everything here
    print(s)
    
