import math
import random
import process_output_HPO   # check that I did this correctly
import sys
import os
import logging

# Finish recursive function, check i=0 submission and save works, parse job ids and test dependency job, sort metrics and find k-highest and submit new job, change directory in i_bracket file, double check correct sorting

def get_hyperparameter_configurations(num_configs, s):
    configs = []
    for x in range(num_configs):
        config_string = 'HPO/bracket_' + str(s) + '/'
        for x in range(len(hyperparameters)):
            config_string += hyperparameters[x] + '_'
            if type(hyperparameter_ranges[x][0]) is float:
                config_string += str(random.SystemRandom().uniform(hyperparameter_ranges[x][0], hyperparameter_ranges[x][1]))
                config_string += '_'
            elif type(hyperparameter_ranges[x][0]) is int:
                config_string += str(random.randrange(hyperparameter_ranges[x][0], hyperparameter_ranges[x][1]))
                config_string += '_'
            elif type(hyperparameter_ranges[x][0]) is str:
                config_string += hyperparameter_ranges[x][random.randrange(len(hyperparameter_ranges[x]))]
                config_string += '_'
            elif type(hyperparameter_ranges[x][0]) is bool:
                config_string += str(random.choice([True, False]))
        #for i in range(1):
        #    configs.append(config_string + '_iteration_' + str(i))
    return configs



# heapSort() helper function
def heapify(metrics, T, n, root):
    largest = root
    left = 2 * root + 1
    right = 2 * root + 2

    # Check if metric[left] exists and is greater than root
    if left < n and metrics[root] < metrics[left]:
        largest = left

    # Check if metric[right] exists and is greater than root
    if right < n and metrics[largest] < metrics[right]:
        largest = right

    # Change root
    if largest != root:
        metrics[root], metrics[largest] = metrics[largest], metrics[root]
        T[root], T[largest] = T[largest], T[root]
        heapify(metrics, T, n, largest)

# Time = nlog(n)
# Space = 1
def heapSort(metrics, T):
    n = len(metrics)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(metrics, T, n, i)

    # swap first and last index, heapify!!
    for i in range(n-1, 0, -1):
        metrics[i], metrics[0] = metrics[0], metrics[i]
        T[i], T[0] = T[0], T[i]
        heapify(metrics, T, i, 0)

    return metrics, T


def recursion(s, i):
    return 0


num_inputs = 25

hyperparameters = ['learning_rate', 'lr_decay_rate', 'width', 'depth', 'activation_function', 'alpha', 'latent_space_size', 'batch_size', 'batch_norm', 'beta1', 'beta2']
hyperparameter_ranges = [[10**-6, 10**-1], [10**-3, 1], [num_inputs, 100], [3, 15], ['Relu', 'sigmoid', 'tanh', 'LeRelu'], [10**-3, 10**-1], [num_inputs, 1000], [500, 20000], [True, False], [0, 1], [0, 1]]

if(len(hyperparameters) != len(hyperparameter_ranges)):
    print("The number of hyperparameters and hyperparameter ranges is not equal")
    sys.exit()

# inputs
R = 3 #27 # maximum number of batches a hyperparameter configuration will run for
eta = 3 # fraction of hyperparameter configuration will be kept (1 / eta), "drop rate"
# initialization
s_max = int(math.floor(math.log(R, eta))) # number of differensftp://awildrid@gilbreth.rcac.purdue.edu/depot/darkmatter/apps/awildrid/HPO/Hyperparameter_Optimization.pyt combinations of total number of configurations and drop rates
B = (s_max + 1) * R # maximum number of batch updates that could be performed with one execution of Hyperband
top_performers = []
for s in [s_max - i for i in range(s_max + 1)]:
    print("s: " + str(s))
    n = int(math.ceil((B / R) * eta**s / (s + 1))) # number of configurations for this iteration
    r = R * eta**(-1 * s) # per bracket base runtime
    # begin successively halving our n configurations with drop rate eta for base runtime r
    T = get_hyperparameter_configurations(n, s)

    # write T to file
    metric_file = open("metric_" + str(s) + "_" + str(i) + ".txt", "w")
    for t in T:
        metric_file.write(t + "\n")
    metric_file.close()


    #call job for first i_bracket
    os.system("python i_bracket.py --i " + str(i) + " --s " + str(s) + " --n " + str(len(T)) + 
              " --r " + str(r) + " --eta " + str(eta) + " --configFile " + "metric_" + str(s) + "_" + str(i) + ".txt") #TODO add arguements


        #top_performer = top_k_performers(T, metric, 1)
    #top_performers.append((top_performer[0][0], top_performer[1][0]))


    
    
    
    