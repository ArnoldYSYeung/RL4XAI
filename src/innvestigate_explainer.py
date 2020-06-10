"""
Implementation of iNNvestigate explainers.

Date:           August 23rd, 2019

See iNNvestigate here:
    https://github.com/albermax/innvestigate

Updates:
    2019-08-27                      Add "one variable" option to switch between tf.keras and keras    
    2019-09-13                      Remove unnecessary print statements
    2019-09-17                      Fix normalize_array() type error 
"""
use_tf = True                   #   whether to use keras or tf.keras

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis

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

from utils import normalize_array

import warnings
warnings.simplefilter('ignore')

import sys
sys.stdout.flush()


def explain_innvestigate(image, label, explainer, cmap_style='plasma', save_name='innvestigate', save_dir=""):
    """
    Creates an explanation using the inputted explainer.
    Arguments:
        image (np.array) :                          array representative of image (width x height x layers)
        label (np.array) :                          prediction of image by classifier (1 x 1)
        explainer (analyzer) :                      explainer to use      
        cmap_style (str) :                          color map code 
        save_name (str) :                           name of file to save output in (ignore extension)
        save_dir (str) :                            directory to save outputs to
    """
    if save_dir != "":
        if save_dir[-1] != "/":
            save_dir += "/"

    #   convert image to 4d
    image = np.array([image])

    """
    print("Image shape: ", image.shape)
    print("Image Pixels: ", np.max(image), np.min(image))
    """

    print("Generating explanations...")
    explanation = explainer.analyze(image).squeeze()        #   width x height x layers
    
    print("Explanation shape: ", explanation.shape)
    print("Explanation Pixels: ", np.max(explanation), np.min(explanation))
    #explanation = ivis.heatmap(explanation)
    """
    #   "overlap" layers
    if explanation.shape[2] > 1:
        explanation = np.sum(explanation, axis=2)
    """
    
    print("Normalizing explanation...")
    #   normalize explanation array
    #explanation = explanation * 100
    explanation = explanation.sum(axis=np.argmax(np.asarray(explanation.shape)==3))         #   convert it to height * width * 1
    if np.max(explanation) != 0:
        explanation /= np.max(np.abs(explanation))
    else:
        print("Warning: Max pixel value is 0.")
    
    max_pixel = np.max(explanation)
    min_pixel = np.min(explanation)
    print("Original Pixels: ", max_pixel, min_pixel)
    explanation = normalize_array(explanation, minmax=[min_pixel, max_pixel], scale_range=[0, 1], dtype=float)
    
    max_pixel = np.max(explanation)
    min_pixel = np.min(explanation)
    print("After Pixels: ", max_pixel, min_pixel)
    print("Explanation shape: ", explanation.shape)
    #print("Range of explanation is : ", max_pixel, min_pixel)
    
    print("Plotting color range...")
    plt.figure()
    color_range = np.array([np.linspace(0, 256, 256)] * 10)
    #print(color_range)
    plt.imshow(color_range, cmap=cmap_style, interpolation='nearest')
    plt.savefig(save_dir+'color_map.png')
    plt.show()
    plt.clf()    
    
    print("Plotting explanation...", end="")
    plt.figure()
    
    plt.imshow(explanation, cmap=cmap_style, interpolation='nearest')
    if save_dir != "":
        if save_dir[-1] != "/":
            save_dir += "/"
        plt.savefig(save_dir+save_name+".png")
    else:
        plt.savefig(save_name+".png")
    print("Saved.")
    plt.show()
    return save_dir+save_name+".png"
    
def experiment(dl_params, model_params, explainer_type, save_dir=""):
    
    keras.backend.clear_session()
    
    #   create data
    print("Loading data...")
    dataloader = Dataloader(dl_params, rseed=0)
    #X_train, y_train = dataloader.get_dataset("train")
    #X_valid, y_valid = dataloader.get_dataset("valid")
    X_test, y_test = dataloader.get_dataset("test")
    del dataloader  # save some memory

    #   convert to np.array
    #X_train = np.stack(X_train, axis=0)
    #X_valid = np.stack(X_valid, axis=0)
    X_test = np.stack(X_test, axis=0)
    #y_train = np.asarray(y_train)
    #y_valid = np.asarray(y_valid)
    y_test = np.asarray(y_test)

    #   normalize to between 0 and 1
    #X_train = X_train.astype("float") / 255.0
    #X_valid = X_valid.astype("float") / 255.0
    X_test = X_test.astype("float") / 255.0

    #image = expand_dims(X_test[0], axis=0)
    image = X_test[70]
    print(image.shape)

    print(matplotlib.get_backend())

    print("Building classifier...")
    #   add this line to prevent some Keras serializer error
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model(model_params['load_location'])
    
    print("Predicting image...")
    label = model.predict(np.array([image,]))
    
    print("The inputted image is predicted to be ", label)

    print("Building explainer...")
    if model_params['output_dim'] > 2:
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)    #   remove softmax
    else:
        model_wo_sm = model
    
    explainer = innvestigate.create_analyzer(explainer_type, model_wo_sm)
    print("Explainer type: ", type(explainer))
    explain_innvestigate(image, label, explainer, save_name=explainer_type, save_dir=save_dir)
    
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
    explainer = "deep_taylor"

    clf_params = {
        'output_dim':           1,
        'activation':           'relu',
        'load_location':        './report/model_e9.hd5'
    }
    save_dir = "./output"

    experiment(dl_params, clf_params, explainer_type=explainer, save_dir=save_dir)