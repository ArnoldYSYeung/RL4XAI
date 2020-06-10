"""
Basic CNN implementation in Keras (with Tensorflow backend)
Upload Date:            April 11th, 2019

Helpful Links:
    Keras CNN with Tensorflow :         https://www.pyimagesearch.com/2018/10/08/keras-vs-tensorflow-which-one-is-better-and-which-one-should-i-learn/
    Keras Callback :                    https://stackoverflow.com/questions/54323960/save-keras-model-at-specific-epochs
    Common CNN Architectures:           https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

Updates:
    2019-04-11                      Initial code - basic structure of CNN class and train() function
    2019-05-04                      Create create_test_report() function
                                        Add classifier testing to experiment()
                                        Add training model save function
    2019-05-05                      Convert Keras to Tensorflow.keras
    2019-05-08                      Modify train() to accept in class Keras.models.Model()
                                        Fix bug in save_n_epoch CustomSaver
    2019-05-13                      Remove label_names parameter from create_test_report()
    2019-08-23                      Convert to Keras model for compatibility
    2019-08-27                      Add "one variable" option to switch between tf.keras and keras
    2019-08-29                      Add Classifier() class from cnn_architectures.py
                                        Add report save location feature
    2019-09-18                      Add training progress saving to txt file in plot_training()
    2019-12-01                      Account for dropout rate bug in tf.keras
"""
use_tf = True

import tensorflow as tf
if use_tf:
    import tensorflow.keras as keras
else:
    import keras
from keras import layers, models, callbacks, optimizers
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import numpy as np
import matplotlib
matplotlib.use("Agg")           #   use .png
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from image_dataloader import Dataloader

import sys
sys.stdout.flush()

class Classifier():
    
    def __init__(self):
        self.model = Model()
        raise NotImplementedError
    
    def convert_to_tensorflow(self, save_dir=None, pb_filename=None, tensorboard=False, input_X=None):
        """
        Convert the existing Keras model to a .pb file (compatiable with pure Tensorflow).
        Arguments:
            save_dir (str) :                directory name to save .pb file into (if None, no saving)
            pb_filename (str) :             filename of .pb file (if None, no saving)
            tensorboard (Boolean) :         whether to plot in tensorboard
        """
        #   get tensorflow session
        
        tf_sess = keras.backend.get_session()
        output_names = [out.op.name for out in self.model.outputs]
        print("Output names: ", output_names)
        
        #print(keras.backend.learning_phase().eval(session=tf_sess))
        
        
        print("Freezing graph...", end="")
        
        #   freeze session and return graph
        #keras.backend.set_learning_phase(0)
        #print("Learning phase 2: ", keras.learning_phase())          #   set tf.keras to inference mode 
        frozen_graph = self._freeze_session(keras.backend.get_session(), output_names=output_names)
        
        if save_dir and pb_filename:
            print("Saving graph...", end="")
            tf.train.write_graph(frozen_graph, save_dir, pb_filename, as_text=False)        #   write to .pb file
            print("DONE.")
        
            #   get graph from file
            from tensorflow.python.platform import gfile
            
            print("save_dir: ", save_dir)
            
            f = gfile.FastGFile(save_dir+"/"+pb_filename, 'rb')
            graph_def = tf.GraphDef()                   #   graphdef class
            graph_def.ParseFromString(f.read())
            f.close()
            tf_sess.graph.as_default()
            tf.import_graph_def(graph_def)          #   this is graph from .pb
        
        if save_dir and pb_filename and tensorboard:
            #   write to tensorboard
            print("Creating file for Tensorboard...")
            writer = tf.summary.FileWriter(save_dir, tf_sess.graph)
            writer.add_graph(tf_sess.graph)
            writer.flush()
            writer.close()
            
            
        #   run tensorflow
        
        #   assume 1 of each
        output_name = self.model.outputs[0].op.name
        input_name = self.model.inputs[0].op.name

        input_X = np.array([input_X])               #   1 * height * width * 3
        
        output_tensor = tf_sess.graph.get_tensor_by_name(output_name+":0")
        prediction = tf_sess.run(output_tensor, {input_name+":0" : input_X})
        print("Hello: ", prediction)

        
        #return tf_sess.graph, output_names
        return tf_sess, output_names
        
    def _freeze_session(self, session, keep_var_names=None, output_names=None, clear_devices=True):
        """
        Freezes the state of a session into a pruned computation graph.
    
        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        
        Code from https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb/
    
        Arguments:
            session (tf.session) :                  The TensorFlow session to be frozen.
            keep_var_names (lst[]) :                A list of variable names that should not be frozen,
                                                      or None to freeze all the variables in the graph.
            output_names (lst[]) :                  Names of the relevant graph outputs.
            clear_devices (Boolean) :               Remove the device directives from the graph for better portability.
        Returns:
            The frozen graph definition. (GraphDef)
        """        
        graph = session.graph
        
        #   initialize all variables so can get graph -- LOL don't do this.  Randomizes weights
        # session.run(tf.global_variables_initializer())
                
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(
                    keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

class CNN(Classifier):
    
    def __init__(self, params):
        
        input_shape = params['input_size']
        num_filters = params['num_filters']
        filter_shape = params['filter_size']
        stride = params['stride']
        padding = params['padding']
        conv_activation = params['conv_activation']
        pool_size = params['pool_size']
        dropout_rate = params['dropout']
        num_conv_layers = params['num_conv_layers']
        output_dim = params['output_dim']
        fc_activation = params['fc_activation']
        
        #   convolutional layers
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        for i in range(1, num_conv_layers+1):
            x = layers.Conv2D(num_filters, filter_shape, strides=stride, 
                       padding=padding)(x)
            x = layers.Activation(conv_activation)(x)
            x = layers.BatchNormalization(axis=-1)(x)              #   batch normalization over depth/channel
            x = layers.MaxPooling2D(pool_size)(x)
            
            if dropout_rate == 1:
                ValueError("Dropout Rate must be less than 1.")
            elif dropout_rate != 0:
                #   bug in tf.keras which sets to 0.5 if dropout_rate = 0 or 1
                x = layers.Dropout(dropout_rate)(x)
            
        #   fully-connected layers
        x = layers.Flatten()(x)
        x = layers.Dense(units=output_dim)(x)
        x = layers.Activation(fc_activation)(x)
        
        #   create model
        self.model = models.Model(inputs, x, name="vanilla_cnn")
        self.model.summary()
        

def train(params, model, X_train, y_train, X_val, y_val):
    """
    Creates the CNN model instance and trains on training data.  Plots results
    of training.  Uses Adam optimizer.
    
    Arguments:
        params (dict) :             experiment parameters
        model (class Model) :       classifier model 
        X_train (np.array) :        training features (num_samples * num_feats)
        y_train (np.array) :        training labels   (num_samples * 1)
        X_val (np.array) :          validation features (num_samples * num_feats)
        y_val (np.array) :          validation labels (num_samples * 1)
    Return:
        clf (class CNN) :           trained classifier  
    """
    
    if 'report_dir' in params and params['report_dir'] != "":
        save_dir = params['report_dir']
        if save_dir[-1] != "/": 
            save_dir += "/"
    else:
        save_dir = ""
    
    #   Callback class for task to be completed at every training epoch
    class CustomSaver(callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (epoch+1) % params['save_n_epoch'] == 0:
                print("Model saved for epoch ", epoch, params['save_n_epoch'])
                self.model.save(save_dir + "model_e{}.h5".format(epoch))
                
    
    lr = params['learning_rate']
    ld = params['learning_decay']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    metrics = params['report_metrics']
    loss_fcn = params['loss_fcn']
    
    print("Learning rate: ", lr)
    print("Learning decay: ", ld)
    
    optimizer = optimizers.Adam(lr=lr, decay=ld)
    model.compile(loss=loss_fcn, optimizer=optimizer, metrics=metrics)
    
    #   train and report loss from validation data
    saver = CustomSaver()
    results = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  batch_size=batch_size, epochs=num_epochs, verbose=2,
                  callbacks=[saver])

    if params['plot_filename'] != "":
        plot_training(params, results, save_dir)
    
    return model
    
def create_test_report(params, y_test, y_pred):
    """
    Create summary of test results and save to file.
    
    Arguments:
        params (dict) :             experiment parameters
        y_test (np.array) :         one-hot label vectors of test data (num_samples * num_classes)
        y_pred (np.array) :         prediction vectors of test data (num_samples * num_classes)
        label_names (list) :        list of names for each label (num_classes)     
    Return:
        report_text (str) :         text of test report
    """
    
    if "report_dir" in params and params['report_dir'] != "":
        save_dir = params['report_dir']
        if save_dir[-1] != "/":
            save_dir += "/"
    else:
        save_dir = ""
    
    filename = save_dir+params['test_filename']

    y_pred = y_pred.flatten()
    y_pred.tolist()
    y_test = y_test.flatten()
    y_test.tolist()
    
    #   convert y_pred to binary 0 and 1
    y_pred = [int(round(x)) for x in y_pred]

    report_text = classification_report(y_test, y_pred)
    
    if filename != "":
        test_report = open(filename, "w")
        test_report.write(report_text)
        test_report.close()
    
    return report_text


def plot_training(params, results, save_dir=""):
    
    num_epochs = params['num_epochs']
    filename = params['plot_filename']
    
    if save_dir != "":
        if save_dir[-1] != "/":
            save_dir += "/"
    
    plt.figure()
    plt.plot(np.arange(0, num_epochs), results.history['acc'], label='train_acc')
    plt.plot(np.arange(0, num_epochs), results.history['val_acc'], label='val_acc')
    plt.plot(np.arange(0, num_epochs), results.history['loss'], label='train_loss')
    plt.plot(np.arange(0, num_epochs), results.history['val_loss'], label='val_loss')
    plt.title("Training Accuracy and Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy / Loss")    
    plt.legend(loc='upper right')
    plt.savefig(save_dir+filename)
    
    txt_file = open(save_dir + "epoch_results.txt", "w+")
    
    #   save text file
    for i in range(0, num_epochs):
        write_string = "epoch: " + str(i) + "    train_acc: " + str(results.history['acc'][i]) + \
        "     train_loss: " + str(results.history['loss'][i]) + "     valid_acc: " + \
        str(results.history['val_acc'][i]) + "      valid_loss: " + str(results.history['val_loss'][i])
        txt_file.write(write_string+"\n")
    txt_file.close()        


def experiment(params, dl_params):
    """
    Runs main pipeline for experiment
    """
    
    #   Load dataset
    print("Loading data...")
    #((X_train, y_train), (X_test, y_test)) = tf.keras.datasets.cifar10.load_data()
    
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

    print(y_train)
    
    print("Building classifier...")
    clf = CNN(params)
    
    print("Training classifier...")
    clf.model = train(params, clf.model, X_train, y_train, X_valid, y_valid)
    del X_train, y_train          #   save memory

    print("Saving classifier...")
    if 'report_dir' in params and params['report_dir'] != "":
        save_dir = params['report_dir']
        if save_dir[-1] != "/": 
            save_dir += "/"
    else:
        save_dir = ""
    keras.backend.set_learning_phase(0)         #   set to inference
    clf.model.save(save_dir + "inference.h5")

    print("Testing classifier...")
    y_pred = clf.model.predict(X_test)
    test_report = create_test_report(params, y_test, y_pred)
    print(test_report)

    keras.backend.clear_session()    
    print("Experiment completed.")
    print("Session ended.")

if __name__ == "__main__":
    
    img_height = 224
    img_width = 224
    img_channels = 3
    num_classes = 1
    
    dl_params = {
            'labels':           ['african', 'indian'],
            'label_type':       'int',
            'file_locs':        ["../data/african_elephant", "../data/indian_elephant"],
            'file_exten':       '.JPEG',
            'set_ratio':        [0.8, 0.1, 0.1],
            'batch_size':       256,
            'target_size':      (224,224),
            'grayscale':        False,
            'superspeedmode':   False                #   trades off memory efficiency for less computation (USE AT YOUR OWN RISK)
    }
    
    
    params = {
            'input_size':           (img_height, img_width, img_channels),
            'num_filters':          3,
            'filter_size':          (5, 5),
            'stride':               1,
            'padding':              'same',
            'conv_activation':      'relu',
            'pool_size':            (2,2),
            'dropout':              0,
            'num_conv_layers':      2,
            'output_dim':           num_classes,
            'fc_activation':        'sigmoid',
            'loss_fcn':             'binary_crossentropy',
            'learning_rate':        0.05,
            'learning_decay':       0.9,
            'num_epochs':           100,
            'batch_size':           256,
            'save_n_epoch':         10,
            'report_metrics':       ['accuracy'],
            'plot_filename':        'training_plot',
            'test_filename':        'test_report',
            'report_dir':           './report'
            }
    
    #densenet_params = params
    #densenet_params['activation'] = 'sigmoid'
    #densenet_params['frozen_layers'] = 300
    
    experiment(params, dl_params)