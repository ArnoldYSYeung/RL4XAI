"""
Runs all the explainers to generate explanations.

Date :                  September 13th, 2019
"""

use_tf = True

if use_tf:
    import tensorflow.keras as keras
else:
    import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

import innvestigate
import innvestigate.utils as iutils

from lime_explainer import explain_lime
from innvestigate_explainer import explain_innvestigate
from protodash_explainer import explain_protodash
from cnn import CNN
from image_dataloader import Dataloader

from datetime import datetime
import numpy as np

def experiment(dl_params, model_params, save_dir):
    """
    Experimental pipeline.
    
    """
    
    #   load dataset
    print("Loading dataset...")
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
    
    #   run Protodash
    print("Running Protodash...")
    print(type(X_train), X_train.shape)
    print(type(X_test), X_test.shape)
    
    proto_indices, weights = explain_protodash((X_train, y_train), (X_test, y_test),
                                               label=None, num_protos=3, 
                                               save_dir=save_dir)
    
    #   get and order samples to explain (most important first)
    samples_to_explain = [idx for _, idx in sorted(zip(weights, proto_indices))]
    samples_to_explain.reverse()
    print("Samples: ", samples_to_explain)
    
    #   load model into classifier
    print("Loading pre-existing classifier...")
    #   add this line to prevent some Keras serializer error
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model(model_params['load_location'])
    
    #   run LIME
    print("Creating LIME explanations...")
    for idx in samples_to_explain:
        #   get image, label corresponding to idx
        image = X_test[idx, :, :, :]
        label = y_test[idx]
        explain_lime(image, label, model, save_name="lime_"+str(idx), save_dir=save_dir)
    
    #   run heatmap
    print("Creating heatmap explanations...")
    #   select analyzer
    explainer_type = "deep_taylor"
    if model_params['output_dim'] > 2:
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)    #   remove softmax
    else:
        model_wo_sm = model
    analyzer = innvestigate.create_analyzer(explainer_type, model_wo_sm)
    
    for idx in samples_to_explain:
        #   get image, label corresponding to idx
        image = X_test[idx, :, :, :]
        label = y_test[idx]
        explain_innvestigate(image, label, analyzer, save_name="heatmap_"+str(idx), save_dir=save_dir)
    

    print("Experiment completed.")

if __name__ == "__main__":
        
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
    
    model_params = {
        'output_dim':           1,
        'activation':           'relu',
        'load_location':        './report/densenet_e9.hd5'
    }
    
    #   create save directory
    save_dir = "./output/"
     
    experiment(dl_params, model_params, save_dir)