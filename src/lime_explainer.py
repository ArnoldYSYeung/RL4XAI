"""
Implementation of LIME for images

Date:           May 14th, 2019

Updates:
    2019-08-23                      Convert to Keras model for compatibility
    2019-08-27                      Add "one variable" option to switch between tf.keras and keras 
    2019-09-13                      Change explain_lime to take in model (instead of classifier)
                                        Remove unnecessary print statements
    2019-09-20                      Modify explain_lime() to have 2 modes (one for complex, large images such as imagenet,
                                            one for simple, small images such as mnist)
"""

use_tf = True

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

import matplotlib
matplotlib.rcParams["backend"] = "PS"
import matplotlib.pyplot as plt

if use_tf:
    import tensorflow.keras as keras
else:
    import keras
from keras.backend import expand_dims
from keras.models import load_model

from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
import numpy as np

from cnn import CNN, train
from cnn_architectures import DenseNet
from image_dataloader import Dataloader


def explain_lime(image, label, model, num_superpixels=10, save_name="lime", save_dir="", imagenet=True):
    """
    Creates an explanation using LIME.
    Arguments:
        image (np.array) :                          array representative of image (width x height x layers)
        label (np.array) :                          prediction of image by classifier (1 x 1)
        model (keras.model) :                       classifier model to explain  
        num_superpixels (int) :                     number of superpixels to highlight
        save_name (str) :                           name of file to save output in (ignore extension)
        save_dir (str) :                            directory to save outputs to
        imagenet (Boolean) :                        mode for more complex, larger images (e.g., imagenet as opposed to mnist)
    Returns:
        list of files (str) of images generated
    """
    
    if save_dir != "":
        if save_dir[-1] != "/":
            save_dir += "/"
            
    print("Creating Explainer...")
    explainer = lime_image.LimeImageExplainer()
    
    if imagenet:
        explanation = explainer.explain_instance(image, model.predict)
        min_weight = 0.0
    else:
        print("Segmenting...")
        #   higher max_dist = fewer clusters
        #   need to have a smaller kernel_size than the default (4) for smaller images
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=255, ratio=0.4)
        print("Explaining...")
        explanation = explainer.explain_instance(image, model.predict, segmentation_fn=segmenter)
        min_weight = 0.05
    
    labels = [0]                #   only 0 if binary
    positive_only = False
    hide_rest = False
    
    #   increase min_weight to the minimum weight seen in the explanation to prevent no superpixels
    exp_max_weight = 0
    for label in labels:
        #   get the maximum weight value in the explanation
        exp = explanation.local_exp[label]
        for f, w in exp[:num_superpixels]:
            if w >= exp_max_weight:
                exp_max_weight = w
    if exp_max_weight < min_weight:
        min_weight = exp_max_weight
    print("Exp_max_weight: ", exp_max_weight)
    print("Min_weight: ", min_weight)
    
    for label in labels:
        filename = plot_mask_for_label(explanation, label, positive_only, hide_rest, num_superpixels, min_weight,
                            save_name=save_name+"_"+str(label), imagenet=imagenet, save_dir=save_dir)
    
    return filename
    
def plot_mask_for_label(explanation, label, positive_only=False, hide_rest=False, 
                        num_superpixels=5, min_weight=0.0, imagenet=True, save_name="lime", save_dir=""):
    """
    Plot mask onto image.
    """
    
    temp, mask = explanation.get_image_and_mask(label, positive_only=positive_only, 
                                                hide_rest=hide_rest, num_features=num_superpixels,
                                                min_weight=min_weight)
    if imagenet:
        plt.imshow(mark_boundaries(temp, mask))
    else:
        #   for mask, 0 = neutral, 1 = class 0, 2 = class 1
        colors = []
        if np.any(mask[:, :] == 1):         #   if 1 in mask
            colors.append('orange')            #   'red' denotes class 1
        if np.any(mask[:, :] == 0):         #   if 2 in mask
            colors.append('magenta')          #   'green' denotes class 2
                
        label_image = label2rgb(mask, temp, colors=colors, alpha=0.5, bg_label=0, kind='overlay')
        plt.imshow(label_image, interpolation="nearest")
    
    if save_dir != "":
        if save_dir[-1] != "/":
            save_dir += "/"
    
    #   save mask to .txt
    np.savetxt(save_dir+save_name+".txt", mask, fmt='%d')
    
    #   save explanation to .png
    plt.savefig(save_dir+save_name+".png")
    plt.show()
    
    return save_dir+save_name+".png"


def experiment(dl_params, model_params, save_dir=""):
    
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
    image = X_test[100]
    print(image.shape)

    print(matplotlib.get_backend())

    print("Building classifier...")
    #clf = DenseNet(model_params)
    model = load_model(model_params['load_location'])
    
    print("Predicting image...")
    label = model.predict(np.array([image,]))
    
    print("The inputted image is predicted to be ", label)

    print("Running LIME...")
    explain_lime(image, label, model, save_dir=save_dir)
    
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
        'load_location':        './report/model_e9.hd5'
    }
    save_dir = "./output"

    experiment(dl_params, clf_params, save_dir)