import numpy as np
import tensorflow.keras as keras
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model

dict_class_num =  {'aeroplane'     :0,
                   'bicycle'       :1,
                   'bird'          :2,
                   'boat'          :3,
                   'bottle'        :4,
                   'bus'           :5,
                   'car'           :6,
                   'cat'           :7,
                   'chair'         :8,
                   'cow'           :9,
                   'dog'           :10,
                   'horse'         :11,
                   'monitor'       :12,
                   'motorbike'     :13,
                   'people'        :14,
                   'pottedplants'  :15,
                   'sheep'         :16,
                   'sofa'          :17,
                   'table'         :18,
                   'trains'        :19}

def matrix_image_loading(path):
    img = Image.open(path)
    img = np.array(img)/255
    return(img)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, path, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path = path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            file_name = ID.split("_")[1]
            
            X[i,] = matrix_image_loading(self.path + file_name)

            # Store class
            y[i] = dict_class_num[self.labels[ID]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        

def net_architecture(path_to_model, base_model_trainable, classes):
    base_model = tf.keras.applications.ResNet50V2(input_shape=(224,224,3),
                                               include_top=False,
                                               weights="imagenet")
    base_model.trainable = base_model_trainable
    new_output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    new_output = tf.keras.layers.Dense(classes,activation='softmax')(new_output)
    model = tf.keras.models.Model(base_model.inputs, new_output)
    if path_to_model != None:
        model.load_weights(path_to_model)
    return(model)

def pascal_experiment(experiment_desc, experiment, path, batch_size, epochs, base_model_trainable, path_to_model):

    if not os.path.exists(experiment_desc):
        os.makedirs(experiment_desc)
    
    dict_class_num =  {'aeroplane'     :0,
                       'bicycle'       :1,
                       'bird'          :2,
                       'boat'          :3,
                       'bottle'        :4,
                       'bus'           :5,
                       'car'           :6,
                       'cat'           :7,
                       'chair'         :8,
                       'cow'           :9,
                       'dog'           :10,
                       'horse'         :11,
                       'monitor'       :12,
                       'motorbike'     :13,
                       'people'        :14,
                       'pottedplants'  :15,
                       'sheep'         :16,
                       'sofa'          :17,
                       'table'         :18,
                       'trains'        :19}

    # Parameters
    params = {'dim': (224,224),
              'batch_size': batch_size,
              'n_classes': 20,
              'n_channels': 3,
              'shuffle': True}

    # Datasets
    data = pickle.load(open(experiment, "rb"))
    data_train, data_validation = train_test_split(list(data), test_size = 0.2)

    partition = {}
    partition['train'] = data_train
    partition['validation'] = data_validation
    labels = {key: value for (key, value) in zip(list(data), [x.split("_")[0] for x in list(data)])}

    # Generators
    training_generator = DataGenerator(partition['train'], labels, path, **params)
    validation_generator = DataGenerator(partition['validation'], labels, path, **params)

    # Design model
    
    optimizer='rmsprop'
    loss='categorical_crossentropy'
    metrics=['accuracy']
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = net_architecture(path_to_model, 
                                 base_model_trainable, 
                                 classes = params["n_classes"])
        
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
    
    model_filename = experiment_desc + "/net_{epoch:02d}_{val_accuracy:.3f}.hdf5"
    
    checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1, save_best_only=False)

    # Train model on dataset
    history = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False, 
                        epochs = epochs,
                        callbacks=[checkpointer], 
                        verbose=1)
    
    history = model.history.history
    pickle.dump(history, open(experiment_desc+"/history_metrics.p", "wb"))
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(experiment_desc+"/acc_curves.jpg")

    with open(experiment_desc + "\summary.txt","w") as text_file:
        print("### Experiment ###", file=text_file)
        print("experiment_desc: {}".format(experiment_desc), file=text_file)
        print("experiment: {}".format(experiment), file=text_file)
        print("path: {}".format(path), file=text_file)
        print("", file=text_file)
        
        print("### Model - training ###", file=text_file)
        print("batch_size: {}".format(batch_size), file=text_file)
        print("epochs: {}".format(epochs), file=text_file)
        print("base_model_trainable: {}".format(base_model_trainable), file=text_file)
        print("path_to_model: {}".format(path_to_model), file=text_file)
        print("", file=text_file)
        
        print("### Model - compilation ###", file=text_file)
        print("optimizer: {}".format(optimizer), file=text_file)
        print("loss: {}".format(loss), file=text_file)
        print("metrics: {}".format(metrics), file=text_file)
    
    return(history)