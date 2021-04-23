import math
import random
import process_output_HPO   # check that I did this correctly
import sys
import os
import logging

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
        for i in range(s):
            configs.append(config_string + '_iteration_' + str(i))
    return configs

def run_then_return_metric(config, r_i, run_num):
    os.system("cd " + config)
    os.system("echo " + str(run_num) + " >> run_num.txt")
    os.system("sbatch -A scholar --nodes=1 --gpus=1 -t 0:05:00 -i ./training_sub_file.sub >> ../output.txt")
    os.system("rm run_num.txt")
    os.system("cd ../")
    #return Process_output_and_metric(testdir = config, run_number = run_num) 

num_inputs = 25

hyperparameters = ['learning_rate', 'lr_decay_rate', 'width', 'depth', 'activation_function', 'alpha', 'latent_space_size', 'batch_size', 'batch_norm', 'beta1', 'beta2']
hyperparameter_ranges = [[10**-6, 10**-1], [10**-3, 1], [num_inputs, 100], [3, 15], ['Relu', 'sigmoid', 'tanh', 'LeRelu'], [10**-3, 10**-1], [num_inputs, 1000], [500, 20000], [True, False], [0, 1], [0, 1]]

if(len(hyperparameters) != len(hyperparameter_ranges)):
    print("The number of hyperparameters and hyperparameter ranges is not equal")
    sys.exit()

# inputs
R = 27 # maximum number of batches a hyperparameter configuration will run for
eta = 3 # fraction of hyperparameter configuration will be kept (1 / eta), "drop rate"
# initialization
s_max = int(math.floor(math.log(R, eta))) # number of different combinations of total number of configurations and drop rates
B = (s_max + 1) * R # maximum number of batch updates that could be performed with one execution of Hyperband
top_performers = []
for s in [s_max - i for i in range(s_max + 1)]:
    print("s: " + str(s))
    n = int(math.ceil((B / R) * eta**s / (s + 1))) # number of configurations for this iteration
    r = R * eta**(-1 * s) # per bracket base runtime
    # begin successively halving our n configurations with drop rate eta for base runtime r
    T = get_hyperparameter_configurations(n, s)

    # make the directories
    for directory in T:
        os.system("mkdir " + directory)
        try:
            os.makedirs(os.path.join("./", directory))
        except OSError:
            logging.warning("Output folders already exist. May overwrite some output files.")

    for i in range(s):
        print("i: " + str(i))
        n_i = int(math.floor(n * eta**(-1 * i)))
        r_i = int(r * eta**i)
        for t in T:
            run_then_return_metric(t, r_i, i)
        #T, metrics = top_k_performers(T, metrics, math.floor(n_i / eta))
    #top_performer = top_k_performers(T, metric, 1)
    #top_performers.append((top_performer[0][0], top_performer[1][0]))


    
    
    
    
