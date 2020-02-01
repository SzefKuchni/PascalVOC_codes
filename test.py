from utils_experiments_multi_gpu import pascal_experiment

experiment_desc = "../../PascalVOC_INSTA_results/local"
experiment = "../exp_50k_no_wrong.pkl"
path = '../../PascalVOC_INSTA_50k/img2/'
batch_size = 4
epochs = 1
path_to_model = None
base_model_trainable = True
l2_regularization = 0.01
learning_rate = 0.001
augment = True

pascal_experiment(experiment_desc, experiment, path, batch_size, epochs, base_model_trainable, path_to_model, l2_regularization, learning_rate, augment)