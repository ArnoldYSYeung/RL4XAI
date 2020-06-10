"""
Train CNN model on Kuzushiji dataset.

Date:                   September 20th, 2019

Updates:
    2019-11-05                      Renamed get_dataset() to get_dataset_images_and_labels()

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
from skimage.color import gray2rgb

from cnn import CNN
from image_dataloader import load_from_npy

import sys
sys.stdout.flush()

def experiment(dl_params, model_params, train_params, train_model=False):

    keras.backend.clear_session() 
    
    #   use gpu
    config = tf.ConfigProto(log_device_placement=True)
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    
    #   save parameters
    print("Saving parameters...")
    save_dir = train_params['report_dir']
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    params_file = open(save_dir + "model_params.txt", "w+")
    params_file.write(str(model_params))
    params_file.close()
    
    params_file = open(save_dir + "train_params.txt", "w+")
    params_file.write(str(train_params))
    params_file.close()
    
    params_file = open(save_dir + "dl_params.txt", "w+")
    params_file.write(str(dl_params))
    params_file.close()
    
    
    #   create data
    print("Loading data...", flush=True)
    dataloader = Dataloader(dl_params, rseed=0)
    X_train, y_train = dataloader.get_dataset_images_and_labels("train")
    X_valid, y_valid = dataloader.get_dataset_images_and_labels("valid")
    X_test, y_test = dataloader.get_dataset_images_and_labels("test")
    
    #dataloader.print_dataset_files(save_dir=save_dir)
    del dataloader      #   save some memory

    #   convert to np.array
    X_train = np.stack(X_train, axis=0)
    X_valid = np.stack(X_valid, axis=0)
    X_test = np.stack(X_test, axis=0)
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)
    y_test = np.asarray(y_test)

    #   normalize to between 0 and 1Convert
    X_train = X_train.astype("float") / 255.0
    X_valid = X_valid.astype("float") / 255.0
    X_test = X_test.astype("float") / 255.0

    #   convert labels to 1-hot vector
    binarizer = LabelBinarizer()
    y_train = binarizer.fit_transform(y_train)
    y_valid = binarizer.fit_transform(y_valid)
    y_test = binarizer.fit_transform(y_test)
    
    #   convert from grayscale to rgb image(LIME requires this...annoying af)
    if X_train.shape[-1] == 1:                       #   if grayscale
        print("Converting from grayscale to RGB...")
        X_train = gray2rgb(X_train.squeeze(axis=-1))
        X_valid = gray2rgb(X_valid.squeeze(axis=-1))
        X_test = gray2rgb(X_test.squeeze(axis=-1))

    print("Building classifier...")
    #   need to add our own "top" FC to make classes=2
    clf = CNN(model_params)

    if train_model is True or model_params['load_location'] == "":
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

    sess.close()
    keras.backend.clear_session()    
    print("Experiment completed.")
    print("Session ended.")
    

if __name__ == "__main__":
    
    
    
    #   check GPU
    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("No GPU.")
    
    num_epochs = 300
    batch_size = 256
    
    dl_params = {
            'labels':           ['a', 'me'],
            'label_type':       'int',
            'presorted':        True,
            'file_locs':        ["../data/kuzushiji-49/0", "../data/kuzushiji-49/33"],
            'file_exten':       '.png',
            #'set_ratio':        [0.7, 0.15, 0.15],
            'set_ratio':        [0.8, 0.2],
            'batch_size':       batch_size,
            'target_size':      (28,28),
            'balanced':         True,
            'grayscale':        True,
            'superspeedmode':   False                #   trades off memory efficiency for less computation (USE AT YOUR OWN RISK)
    }
    
    train_params = {
            'learning_rate':        0.0075,
            'learning_decay':       0.9,
            'num_epochs':           num_epochs,
            'batch_size':           batch_size,
            'save_n_epoch':         10,
            'loss_fcn':             'binary_crossentropy',
            'report_metrics':       ['accuracy'],
            'plot_filename':        'training_plot',
            'test_filename':        'test_report',
            'report_dir':           './report/',
            'report_metrics':       ['accuracy'],
            }

    
    img_height, img_width, img_channels = (28, 28, 3)
    num_classes = 1             #   use 1 if binary
    
    cnn_params = {
            'input_size':           (img_height, img_width, img_channels),
            'num_filters':          3,
            'filter_size':          (5, 5),
            'stride':               1,
            'padding':              'same',
            'conv_activation':      'relu',
            'pool_size':            (2,2),
            'dropout':              0.1,
            'num_conv_layers':      2,
            'output_dim':           num_classes,
            'fc_activation':        'sigmoid'
            }
    
    
    experiment(dl_params, cnn_params, train_params, train_model=True)