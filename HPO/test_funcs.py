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
            configs.append(config_string + 'iteration_' + str(i))
    return configs

print(get_hyperparameter_configurations(15, 4))