import numpy as np
import tensorflow.keras as keras
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa

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

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, path, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, augment = False, scale = (0.8, 1.2), translate_percent = (-0.2, 0.2), rotate = (-45, 45), shear = (-16,16)):
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
        self.augment = augment
        self.scale = scale
        self.translate_percent = translate_percent
        self.rotate = rotate
        self.shear = shear

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
            
            X[i,] = self.matrix_image_loading(self.path + file_name)

            # Store class
            y[i] = dict_class_num[self.labels[ID]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        
    def matrix_image_loading(self, path):
        img = Image.open(path)
        img = np.array(img)#/255
        if self.augment == True:
           img = self.augmentor(img)
        img = np.array(img)/255
        return img
        
    def augmentor(self, images):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
                [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                #iaa.Flipud(0.2),  # vertically flip 20% of all images
                sometimes(iaa.Affine(
                    scale={"x": self.scale, "y": self.scale},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": self.translate_percent, "y": self.translate_percent},
                    # translate by -20 to +20 percent (per axis)
                    rotate=self.rotate,  # rotate by -45 to +45 degrees
                    shear=self.shear,  # shear by -16 to +16 degrees
                    order=[0, 1],
                    # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                ],
                random_order=True
        )
        return seq.augment_image(images)
        

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

def pascal_experiment(experiment_desc, experiment, path, batch_size, epochs, base_model_trainable, path_to_model, l2_regularization, learning_rate, augment, scale, translate_percent, rotate, shear):

    if not os.path.exists(experiment_desc):
        os.makedirs(experiment_desc)

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
    training_generator = DataGenerator(partition['train'], labels, path, augment = augment, scale = scale, translate_percent = translate_percent, rotate = rotate, shear = shear, **params)
    validation_generator = DataGenerator(partition['validation'], labels, path, **params)

    # Design model
    
    optimizer= keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss='categorical_crossentropy'
    metrics=['accuracy']
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = net_architecture(path_to_model, 
                                 base_model_trainable, 
                                 classes = params["n_classes"])
                                 
        for layer in model.layers:
            condition = layer.get_config()["name"].split("_")[-1]
            if condition == "conv":
                layer.kernel_regularizer = keras.regularizers.l2(l2_regularization)
            else:
                continue
        
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

    with open(experiment_desc + "/summary.txt","w") as text_file:
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
        print("learning_rate: {}".format(learning_rate), file=text_file)
        print("l2_regularization: {}".format(l2_regularization), file=text_file)
        print("augment: {}".format(augment), file=text_file)
        print("scale: {}".format(scale), file=text_file)
        print("translate_percent: {}".format(translate_percent), file=text_file)
        print("rotate: {}".format(rotate), file=text_file)
        print("shear: {}".format(shear), file=text_file)
        print("", file=text_file)
        
        print("### Model - compilation ###", file=text_file)
        print("optimizer: {}".format(optimizer), file=text_file)
        print("loss: {}".format(loss), file=text_file)
        print("metrics: {}".format(metrics), file=text_file)
    
    return(history)