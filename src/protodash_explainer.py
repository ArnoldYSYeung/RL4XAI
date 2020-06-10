"""
Implementation of ProtoDash explainer.  Currently only selects prototypes, no
criticisms :(

Date:           September 12th, 2019

See AIX360 here:
    https://github.com/IBM/AIX360

Updates:
    2019-09-17                              Add filename arguments to explain_protodash
                                            Add report .txt file
"""
use_tf = True                   #   whether to use keras or tf.keras

from aix360.algorithms.protodash import ProtodashExplainer, get_Gaussian_Data

import matplotlib
matplotlib.rcParams["backend"] = "PS"
import matplotlib.pyplot as plt

if use_tf:
    import tensorflow.keras as keras
else:
    import keras
from keras.backend import expand_dims
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

from skimage.segmentation import mark_boundaries
import numpy as np

from cnn import CNN, train
from cnn_architectures import DenseNet
from image_dataloader import Dataloader

from utils import plot_numpy_images

import warnings
warnings.simplefilter('ignore')

import sys, os
sys.stdout.flush()


def explain_protodash(dataset, select_dataset,
                      label=None, num_protos=10, filename="", mark_label=False, 
                      label_names=[], font_file="", save_dir=""):
    """
    Select prototypes and their associated weights using the Protodash explainer.
    Arguments:
        dataset (tuple()) : 
            (np.array) :                  dataset for prototypes to represent (target)
            (np.array) :                  labels corresponding to dataset
        select_dataset (tuple()) :
            (np.array) :                dataset containing samples for prototype selection (source)
            (np.array) :                labels corresponding to selection dataset
        label (int) :                   label to find prototypes for (if None, all labels)
        num_protos (int) :              number of prototypes to select
        filename (str) :                filename of output images (if "", images not saved)
        mark_label (Boolean) :          whether to caption labels
        label_names (lst[str, ]) :      names of labels
        font_file (str) :               font file for label captions
        save_dir (str) :                directory to save prototypes to
    """
    
    num_samples, height, width, num_layers = dataset[0].shape
    select_num_samples, select_height, select_width, select_num_layers = select_dataset[0].shape
    
    #   check that label exists in datasets
    if label is not None:
        if label not in dataset[0]:
            raise ValueError("Dataset does not contain label.")
        if label not in select_dataset[0]:
            raise ValueError("Selection dataset does not contain label.")
    
    if height != select_height or width != select_width or num_layers != select_num_layers:
        raise Exception('Dimensions of datasets are mismatched.')
    
    if num_samples != dataset[1].shape[0]:
        print("Labels: ", dataset[1].shape[0])
        print("Num samples: ", num_samples)
        raise Exception("Label size does not match dataset.")
    
    if select_num_samples != select_dataset[1].shape[0]:
        print("Labels: ", select_dataset[1].shape[0])
        print("Num samples: ", num_samples)
        raise Exception("Label size does not match selection dataset.")
    
    if select_num_samples < num_samples:
        print("Warning: Number of samples in selection dataset is smaller than in dataset to represent.")
    
    print("Flattening data...")
    #   reshape to 2d (num_samples * (height*width*num_layers))
    dataset2d = dataset[0].reshape((num_samples, height*width*num_layers))
    select_dataset2d = select_dataset[0].reshape((select_num_samples, 
                                                     select_height*select_width*
                                                     select_num_layers))

    #   only keep data of the matching label
    if label is not None:
        print("Finding prototypes for only label", label)
        label_indices = np.where(dataset[1] == label)
        dataset2d = dataset2d[label_indices]
        select_label_indices = np.where(select_dataset[1] == label)[0]
        select_dataset2d = select_dataset2d[select_label_indices]

    print("Explaining...")
    explainer = ProtodashExplainer()
    
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')      #   suppress printing
    weights, proto_indices, _ = explainer.explain(dataset2d, select_dataset2d, 
                                                        num_protos, kernelType='other')
    sys.stdout.close()
    sys.stdout = original_stdout
    
    #   convert proto_indices to match indices of original original dataset
    if label is not None:
        proto_indices = [select_label_indices[i] for i in proto_indices]
    
    #   normalize weights
    weights = np.around(weights/np.sum(weights), 2)
    
    #   order prototype indices based on highest weights
    proto_indices = [idx for _, idx in sorted(zip(weights, proto_indices))]
    
    #   plot if only 1 image
    if num_samples == 1:
        print("Plotting reference image...")
        output_name = plot_numpy_images(dataset, filename=filename, save_dir=save_dir)
    #   plot prototypes
    print("Plotting prototypes...")
    prototypes = select_dataset[0][proto_indices]
    prototype_labels = select_dataset[1][proto_indices]
    print("Number of prototypes: ", prototypes.shape)
    if filename != "":    
        output_name = plot_numpy_images((prototypes, prototype_labels), filename=filename, 
                          indices=proto_indices, mark_label=mark_label, label_names=label_names, 
                          font_file=font_file, concat=True, save_dir=save_dir)
    
    #   report the prototype indices and their corresponding weights
    f = open(save_dir+filename+".txt", "w+")
    for i in range(0, len(proto_indices)):
        write_string = "Index: " + str(proto_indices[i]) + \
                        "    Weight: " + str(weights[i]) + "\n"
        f.write(write_string)
    f.close()
    
    del explainer
    del dataset2d, select_dataset2d, dataset, select_dataset, prototypes
    
    return proto_indices, weights, output_name
    
def experiment(dl_params, model_params, label=None, num_protos=10, save_dir=""):
    
    keras.backend.clear_session()
    
    #   create data
    print("Loading data...")
    dataloader = Dataloader(dl_params, rseed=0)
    X_train, y_train = dataloader.get_dataset("train")
    X_valid, y_valid = dataloader.get_dataset("valid")
    X_test, y_test = dataloader.get_dataset("test")
    del dataloader  # save some memory

    #   convert to np.array
    X_train = np.stack(X_train, axis=0)
    #X_valid = np.stack(X_valid, axis=0)
    X_test = np.stack(X_test, axis=0)
    y_train = np.asarray(y_train)
    #y_valid = np.asarray(y_valid)
    y_test = np.asarray(y_test)

    #   normalize to between 0 and 1
    X_train = X_train.astype("float") / 255.0
    #X_valid = X_valid.astype("float") / 255.0
    X_test = X_test.astype("float") / 255.0

    #image = expand_dims(X_test[0], axis=0)
    image = np.array([X_test[70]])
    image_label = np.array([label])
    print(image.shape)

    print(matplotlib.get_backend())

    print(image.shape)
    print(image_label.shape)

    #   single image
    proto_indices, weights = explain_protodash((image, image_label), (X_train, y_train),
                                               label=label, num_protos=num_protos, 
                                               save_dir=save_dir)
    """
    #   multiple images
    proto_indices, weights = explain_protodash((X_train, y_train), (X_train, y_train),
                                               label=label, num_protos=num_protos, 
                                               save_dir=save_dir)
    """
    print("Prototype Indices: ", proto_indices)
    print("Weights: ", weights)
    
    keras.backend.clear_session()

if __name__ == "__main__":

    print("Loading parameters...")
    matplotlib.rcParams["backend"] = "TkAgg"
    
    dl_params = {
        'labels': ['tabby', 'siamese'],
        'label_type': 'int',
        'file_locs': ["../data/tabby_cat", "../data/siamese_cat"],
        'file_exten': '.JPEG',
        'set_ratio': [0.8, 0.1, 0.1],
        'batch_size': 32,
        'target_size': (224, 224),
        'superspeedmode': False  # trades off memory efficiency for less computation (USE AT YOUR OWN RISK)
    }

    img_height = 32
    img_width = 32
    img_channels = 3
    num_classes = 2

    clf_params = {
        'output_dim':           1,
        'activation':           'relu',
        'load_location':        'model_e9.hd5'
    }
    save_dir = "./output"
    label = 0
    num_protos = 10
    experiment(dl_params, clf_params, label=label, num_protos=num_protos,
               save_dir=save_dir)