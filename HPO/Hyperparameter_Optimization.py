import math
import random

num_inputs = 25

hyperparameters = ['learning_rate', 'lr_decay_rate', 'width', 'depth', 'activation_function', 'alpha', 'latent_space_size', 'batch_size', 'batch_norm']
hyperparameter_ranges = [[10**-6, 10**-1], [10**-3, 1], [num_inputs, 100], [3, 15], ['Relu', 'sigmoid', 'tanh', 'LeRelu'], [10**-3, 10**-1], [num_inputs, 1000], [500, 20000], [True, False]]

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

def top_k_performers(T, metrics, k):
    metrics, T = heapSort(metrics, T)

    return T[:k], metrics[:k]

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


if(len(hyperparameters) != len(hyperparameter_ranges)):
    print("The number of hyperparameters and hyperparameter ranges is not equal")
    sys.exit()

# inputs
R = 177147 # maximum number of batches a hyperparameter configuration will run for
eta = 3 # fraction of hyperparameter configuration will be kept (1 / eta), "drop rate"
# initialization
s_max = math.floor(math.log(R, eta)) # number of different combinations of total number of configurations and drop rates
B = (s_max + 1) * R # maximum number of batch updates that could be performed with one execution of Hyperband
top_performers = []
for s in [s_max - i for i in range(s_max + 1)]:
    print("s: " + str(s))
    n = math.ceil((B / R) * eta**s / (s + 1)) # number of configurations for this iteration
    r = R * eta**(-1 * s) # per bracket base runtime
    # begin successively halving our n configurations with drop rate eta for base runtime r
    T = get_hyperparameter_configurations(n, s)
    for i in range(s):
        print("i: " + str(i))
        n_i = math.floor(n * eta**(-1 * i))
        r_i = r * eta**i
        metrics = [run_then_return_metric(t, r_i) for t in T]
        T, metrics = top_k_performers(T, metrics, math.floor(n_i / eta))
    top_performer = top_k_performers(T, metric, 1)
    top_performers.append((top_performer[0][0], top_performer[1][0]))
