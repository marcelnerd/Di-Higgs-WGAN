#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--num", type=int, help="integer which tells iteration")
opt = parser.parse_args()
i = opt.num

#print(str(i ** 2) + " on the {j}th iteration".format(j = i))
np.save("./" + '{j}.npy'.format(j = i), 2*i)

