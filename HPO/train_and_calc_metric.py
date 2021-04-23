#!/usr/bin/env python
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, help="path to directory")
opt = parser.parse_args()
i = opt.dir
print(i)

#print(str(i ** 2) + " on the {j}th iteration".format(j = i))
#np.save("./" + '{j}.npy'.format(j = i), 2*i)