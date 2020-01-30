from utils_experiments_multi_gpu import pascal_experiment

experiment_desc = "../results/multi_gpu"
experiment = "../experiments/exp_50k_no_wrong.pkl"
path = '../PascalVOC/img/'
batch_size = 64*4
epochs = 20
path_to_model = None
base_model_trainable = True

pascal_experiment(experiment_desc, experiment, path, batch_size, epochs, base_model_trainable, path_to_model)