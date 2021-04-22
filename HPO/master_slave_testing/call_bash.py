#!/usr/bin/env python
import subprocess
import os
import numpy as np
import time

for s in range(0,3):
    jobIDs = ""
    #os.system("rm output.txt")
    for i in range(0,4):
        
        os.system("sbatch -A scholar" + " --nodes=1 --gpus=1 -t 1:00:00 ./submission.sub --i " + \
                         str(i) + ">> output" + str(i) + ".txt") #; wait 2")
        
        #time.sleep(0.01)
        #os.system("cat output.txt")
    #os.system("cat output.txt")
    #file = open("output.txt","r") 
   # line = file.readline()
    #count = 0
    #while line:
     #   jobIDs += line.split(" ")[3].rstrip("\n")
        #if (count < 4 - 1):
         #   jobIDs += ":"
      #  line = file.readline()
    #print(jobIDs)
    #file.close()
    os.system("sbatch -A scholar --nodes=1 --gpus=1 -t 1:00:00 --dependency=afterok:" + jobIDs + " ./empty.sub")
    #load saved stuff and sort everything here
    print(s)
    
