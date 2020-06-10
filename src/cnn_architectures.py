"""
Application of different CNN architectures using Keras/Tf.Keras
    - DenseNet

Upload Date:            May 8th, 2019

Updates:
    2019-08-23                      Convert to Keras model for compatibility
    2019-08-27                      Add "one variable" option to switch between tf.keras and keras
    2019-08-28                      Add convert_to_tensorflow() for classifiers
                                            - based on https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a
    2019-08-29                      Moved Classifier() class to cnn.py                                        
"""
use_tf = True

from image_dataloader import Dataloader 
from cnn import train, create_test_report, Classifier

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

if use_tf:
    import tensorflow.keras as keras
else:
    import keras
import tensorflow as tf
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from sklearn.preprocessing import LabelBinarizer

import sys
sys.stdout.flush()

class DenseNet(Classifier):

    def __init__(self, params, verbose=False):

        output_size = params['output_dim']
        activation = params['activation']
        freeze_num_layers = params['frozen_layers']

        base_model = DenseNet121(weights='imagenet', include_top=False)
        x = base_model.output
        #   fully connected layer
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation=activation)(x)
        out = Dense(output_size, activation='sigmoid')(x)

        self.model = Model(inputs=base_model.input, outputs=out)

        #   set trainability of layers
        print("Number of layers: ", len(base_model.layers))
        if freeze_num_layers > 0:
            for layer in base_model.layers[:freeze_num_layers]:                   #   freeze layers
                layer.trainable = False
            for layer in base_model.layers[freeze_num_layers:]:
                layer.trainable = True

        if verbose == True:
            self.model.summary()

        #   compile model - in train()
        #   self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')


def experiment(dl_params, model_params, train_params, train_model=False):

    #   create data
    print("Loading data...", flush=True)
    dataloader = Dataloader(dl_params, rseed=0)
    X_train, y_train = dataloader.get_dataset("train")
    X_valid, y_valid = dataloader.get_dataset("valid")
    X_test, y_test = dataloader.get_dataset("test")
    del dataloader      #   save some memory

    #   convert to np.array
    X_train = np.stack(X_train, axis=0)
    X_valid = np.stack(X_valid, axis=0)
    X_test = np.stack(X_test, axis=0)
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)
    y_test = np.asarray(y_test)

    #   normalize to between 0 and 1
    X_train = X_train.astype("float") / 255.0
    X_valid = X_valid.astype("float") / 255.0
    X_test = X_test.astype("float") / 255.0

    #   convert labels to 1-hot vector
    binarizer = LabelBinarizer()
    y_train = binarizer.fit_transform(y_train)
    y_valid = binarizer.fit_transform(y_valid)
    y_test = binarizer.fit_transform(y_test)

    print("Building classifier...")
    #   need to add our own "top" FC to make classes=2
    clf = DenseNet(model_params)

    if train_model is True:
        print("Training classifier...")
        clf.model = train(train_params, clf.model, X_train, y_train, X_valid, y_valid)
    elif train_model is False:
        clf.model = load_model(model_params['load_location'])
    else:
        pass            #   use untrained model
    del X_train, X_valid, y_train, y_valid          #   save memory
    
    print("Testing classifier...")
    y_pred = clf.model.predict(X_test)

    test_report = create_test_report(train_params, y_test, y_pred)
    print(test_report)

    keras.backend.clear_session()    
    print("Experiment completed.")
    print("Session ended.")
    

if __name__ == "__main__":
    
    #   check GPU
    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("No GPU.")
    
    num_epochs = 10
    batch_size = 256
    
    dl_params = {
            'labels':           ['tabby', 'siamese'],
            'label_type':       'int',
            'file_locs':        ["../data/tabby_cat", "../data/siamese_cat"],
            'file_exten':       '.JPEG',
            'set_ratio':        [0.8, 0.1, 0.1],
            'batch_size':       batch_size,
            'target_size':      (224,224),
            'superspeedmode':   False                #   trades off memory efficiency for less computation (USE AT YOUR OWN RISK)
    }
    
    train_params = {
            'learning_rate':        0.000075,
            'learning_decay':       0.9,
            'num_epochs':           num_epochs,
            'batch_size':           batch_size,
            'save_n_epoch':         100,
            'loss_fcn':             'binary_crossentropy',
            'report_metrics':       ['accuracy'],
            'plot_filename':        'training_plot',
            'test_filename':        'test_report'
            }

    model_params = {
        'output_dim':               1,
        'activation':               'relu',
        'frozen_layers':            300,
        'load_location':            ''
    }
    
    experiment(dl_params, model_params, train_params, train_model=True)