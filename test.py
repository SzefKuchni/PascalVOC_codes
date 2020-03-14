from utils_experiments_multi_gpu import pascal_experiment

# Global arguments
experiment = "../experiments/exp_50k_no_wrong.pkl"
path = '../PascalVOC/img/'
batch_size = 4*64
epochs = 20
path_to_model = None
base_model_trainable = True

# EXPERIMENT 1
# Local arguments
experiment_desc = "../results/mgpu_l2_005"
l2_regularization = 0.005
learning_rate = 0.001
augment = False
scale = (0.8, 1.2)
translate_percent = (-0.2, 0.2)
rotate = (-45, 45)
shear = (-16,16)

pascal_experiment(  experiment_desc, 
                    experiment, 
                    path, 
                    batch_size, 
                    epochs, 
                    base_model_trainable, 
                    path_to_model, 
                    l2_regularization, 
                    learning_rate, 
                    augment, 
                    scale, 
                    translate_percent, 
                    rotate, 
                    shear)
                    
# EXPERIMENT 2
# Local arguments
experiment_desc = "../results/mgpu_l2_05"
l2_regularization = 0.05
learning_rate = 0.001
augment = False
scale = (0.8, 1.2)
translate_percent = (-0.2, 0.2)
rotate = (-45, 45)
shear = (-16,16)

pascal_experiment(  experiment_desc, 
                    experiment, 
                    path, 
                    batch_size, 
                    epochs, 
                    base_model_trainable, 
                    path_to_model, 
                    l2_regularization, 
                    learning_rate, 
                    augment, 
                    scale, 
                    translate_percent, 
                    rotate, 
                    shear)
                    
# EXPERIMENT 3
# Local arguments
experiment_desc = "../results/mgpu_l2_001_DA_rotate_15_scale_02_translate_02"
l2_regularization = 0.01
learning_rate = 0.001
augment = True
scale = (0.8, 1.2)
translate_percent = (-0.2, 0.2)
rotate = (-15, 15)
shear = (-16,16)

pascal_experiment(  experiment_desc, 
                    experiment, 
                    path, 
                    batch_size, 
                    epochs, 
                    base_model_trainable, 
                    path_to_model, 
                    l2_regularization, 
                    learning_rate, 
                    augment, 
                    scale, 
                    translate_percent, 
                    rotate, 
                    shear)
                    
# EXPERIMENT 4
# Local arguments
experiment_desc = "../results/mgpu_l2_001_DA_rotate_30_scale_02_translate_02"
l2_regularization = 0.01
learning_rate = 0.001
augment = True
scale = (0.8, 1.2)
translate_percent = (-0.2, 0.2)
rotate = (-30, 30)
shear = (-16,16)

pascal_experiment(  experiment_desc, 
                    experiment, 
                    path, 
                    batch_size, 
                    epochs, 
                    base_model_trainable, 
                    path_to_model, 
                    l2_regularization, 
                    learning_rate, 
                    augment, 
                    scale, 
                    translate_percent, 
                    rotate, 
                    shear)
                    
# EXPERIMENT 5
# Local arguments
experiment_desc = "../results/mgpu_l2_001_DA_rotate_45_scale_02_translate_02"
l2_regularization = 0.01
learning_rate = 0.001
augment = True
scale = (0.8, 1.2)
translate_percent = (-0.2, 0.2)
rotate = (-45, 45)
shear = (-16,16)

pascal_experiment(  experiment_desc, 
                    experiment, 
                    path, 
                    batch_size, 
                    epochs, 
                    base_model_trainable, 
                    path_to_model, 
                    l2_regularization, 
                    learning_rate, 
                    augment, 
                    scale, 
                    translate_percent, 
                    rotate, 
                    shear)
                    
# EXPERIMENT 6
# Local arguments
experiment_desc = "../results/mgpu_l2_001_DA_rotate_30_scale_01_translate_02"
l2_regularization = 0.01
learning_rate = 0.001
augment = True
scale = (0.9, 1.1)
translate_percent = (-0.2, 0.2)
rotate = (-30, 30)
shear = (-16,16)

pascal_experiment(  experiment_desc, 
                    experiment, 
                    path, 
                    batch_size, 
                    epochs, 
                    base_model_trainable, 
                    path_to_model, 
                    l2_regularization, 
                    learning_rate, 
                    augment, 
                    scale, 
                    translate_percent, 
                    rotate, 
                    shear)
                    
# EXPERIMENT 7
# Local arguments
experiment_desc = "../results/mgpu_l2_001_DA_rotate_30_scale_03_translate_02"
l2_regularization = 0.01
learning_rate = 0.001
augment = True
scale = (0.7, 1.3)
translate_percent = (-0.2, 0.2)
rotate = (-30, 30)
shear = (-16,16)

pascal_experiment(  experiment_desc, 
                    experiment, 
                    path, 
                    batch_size, 
                    epochs, 
                    base_model_trainable, 
                    path_to_model, 
                    l2_regularization, 
                    learning_rate, 
                    augment, 
                    scale, 
                    translate_percent, 
                    rotate, 
                    shear)
                    
# EXPERIMENT 8
# Local arguments
experiment_desc = "../results/mgpu_l2_001_DA_rotate_30_scale_02_translate_01"
l2_regularization = 0.01
learning_rate = 0.001
augment = True
scale = (0.2, 1.2)
translate_percent = (-0.1, 0.1)
rotate = (-30, 30)
shear = (-16,16)

pascal_experiment(  experiment_desc, 
                    experiment, 
                    path, 
                    batch_size, 
                    epochs, 
                    base_model_trainable, 
                    path_to_model, 
                    l2_regularization, 
                    learning_rate, 
                    augment, 
                    scale, 
                    translate_percent, 
                    rotate, 
                    shear)
                    
# EXPERIMENT 9
# Local arguments
experiment_desc = "../results/mgpu_l2_001_DA_rotate_30_scale_02_translate_03"
l2_regularization = 0.01
learning_rate = 0.001
augment = True
scale = (0.2, 1.2)
translate_percent = (-0.3, 0.3)
rotate = (-30, 30)
shear = (-16,16)

pascal_experiment(  experiment_desc, 
                    experiment, 
                    path, 
                    batch_size, 
                    epochs, 
                    base_model_trainable, 
                    path_to_model, 
                    l2_regularization, 
                    learning_rate, 
                    augment, 
                    scale, 
                    translate_percent, 
                    rotate, 
                    shear)