"""
Dataloader for Imagenet dataset

Upload Date:            May 5th, 2019

Useful Links:
    Load Images into Keras :                https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/
                                            https://github.com/fchollet/deep-learning-models
                                            
Updates:
    2019-05-07                          Add get_num_batches() and minor updates to other functions
                                            Add in superspeedmode (trades memory efficiency for less computations, USE AT OWN RISK)
                                            Add in get_dataset() function
                                            Change output of load_batch and get_dataset to be 2 np.arrays (X and y)
                                            Add target_size for get_image_array()
    2019-05-16                          Add plot_image()
    2019-08-23                          Convert to Keras model for compatibility
    2019-08-27                          Add "one variable" option to switch between tf.keras and keras
    2019-09-20                          Add save_npy_as_images()
                                            Add Dataloader.print_dataset_files()
    2019-10-03                          Add feature for already sorted datasets (train, valid, test)
    2019-11-05                          Add get dataset_names_and_labels()
                                            Renamed get_dataset() to get_dataset_images_and_labels()
    2019-11-29                          Add in (training and test) dataset balancing option for presorted option
"""
use_tf = True
if use_tf:
    import tensorflow.keras as keras
else:
    import keras

from keras.preprocessing import image

import numpy as np

import os
from os import listdir
from os.path import splitext
import random
import math
import matplotlib.pyplot as plt
import itertools

class Dataloader():
    """
    Dataloader class for image data stored as .png, etc.
    """
    
    def __init__(self, params, rseed = None, verbose=True):
        """
        Create Dataloader instance.  Extract filenames and separate into training, validation,
        and test sets.
        
        Arguments:
            params (dict) :         contains parameters of dataloader
            rseed  (int) :          random seed (set to None for no random seed)
            verbose (boolean) :     print messages or not
        """
        if rseed != None:
            random.seed(rseed)
        
        self.labels = params['labels']
        self.file_locs = params['file_locs']
        self.presorted = params['presorted']                                    #   if training and test files are sorted into separate folders
        batch_size = params['batch_size']
        self.target_size = params['target_size']
        self.superspeedmode = params['superspeedmode']
        self.grayscale = params['grayscale']
        
        self.classes = range(0, len(self.labels))                               #   list of class starting at 0
        self.num_batches = {'train': 0, 'valid': 0, 'test': 0}                  #   contains total num of batches in each set

        extension = params['file_exten']
        ratios = params['set_ratio']
        
        #   calculate dataset ratios
        if self.presorted is False:
            if len(ratios) != 3:
                raise ValueError("Ratio does not have 3 values.")
            train_percent = ratios[0] / sum(ratios)
            valid_percent = ratios[1] / sum(ratios)
            test_percent = ratios[2] / sum(ratios) 
        else:
            if len(ratios) != 2:
                raise ValueError("Ratio does not have 2 values.")
            train_percent = ratios[0] / sum(ratios)
            valid_percent = ratios[1] / sum(ratios)
        
        files = {}
        file_storage = {'train': [], 'valid': [], 'test': []}
        label_storage = {'train': [], 'valid': [], 'test': []}
        loc_dataset = {}
        self.X_batches = {}
        self.y_batches = {}
        
        count_label = 0             #   used for integer labels
        
        if params['balanced']:
            #   find minimum number of training files in a set
            min_num_train, min_num_test = None, None
            min_train_class, min_test_class = None, None
            for index in range(0, len(self.file_locs)):
                train_folder = self.file_locs[index]+"/train/"          #   folder for each class
                test_folder = self.file_locs[index]+"/test/"          #   folder for each class
                #   number of training files
                num_train = len([f for f in os.listdir(train_folder) if 
                                 f.endswith(extension) and os.path.isfile(
                                         os.path.join(train_folder, f))])
                num_test = len([f for f in os.listdir(test_folder) if 
                                 f.endswith(extension) and os.path.isfile(
                                         os.path.join(test_folder, f))])
                if min_num_train is None or num_train < min_num_train:
                    min_num_train = num_train
                    min_train_class = index
                if min_num_test is None or num_test < min_num_test:
                    min_num_test = num_test
                    min_test_class = index
        
        #   assign train, valid, test files for each class
        for index in range(0, len(self.file_locs)):   
            folder = self.file_locs[index]          #   folder for each class
            
            if params['label_type'] == 'int':       #   use integer labels
                label = count_label
                count_label += 1
            else:           #   use labels in params['labels']
                label = self.labels[index]
            
            if verbose is True:
                print("Loading data from", folder, "...")
            
            #   if not pre-sorted into train, valid, test datasets
            if self.presorted is False:
                #   get filenames in each directory (class)
                files[label] = [str(folder + '/' + f) for f in listdir(folder) if splitext(f)[1] == extension]
                
                num_files = len(files[label])
                if verbose is True:
                    print("There are", num_files, "files. ")
                if num_files <= 0:
                    raise Exception("No " + extension + " files found.  Try changing the file extension type.")
                
                #   shuffle filenames
                random.shuffle(files[label])
                
                #   separate training from validation into separate list
                num_train = int(num_files * train_percent)
                num_valid = int(num_files * valid_percent)
                num_test = int(num_files * test_percent)
                
                file_storage['train'] += files[label][:num_train]
                label_storage['train'] += [label] * num_train
                file_storage['valid'] += files[label][num_train:num_train+num_valid]
                label_storage['valid'] += [label] * num_valid
                file_storage['test'] += files[label][num_train+num_valid:num_train+num_valid+num_test]
                label_storage['test'] += [label] * num_test
            
                #   file_storage contains the locations (string) of each file
                #   label_storage contains the labels (int) of each file
            
            #   if already pre-sorted in /train/, /test/ folders
            #   if pre-sorted is True, then ratios are not used to calculate sort datasets
            #   valid data is taken from /train/
            else:
                
                train_folder = str(folder)+'/train/'
                test_folder = str(folder)+'/test/'
                
                #   check if folder exists
                if os.path.isdir(train_folder) is False:
                    raise ValueError(str(train_folder) + " does not exist.")
                elif os.path.isdir(test_folder) is False:
                    raise ValueError(str(test_folder) + " does not exist.")
                
                train_files = [train_folder + f for f in listdir(train_folder) if splitext(f)[1] == extension]
                test_files = [test_folder + f for f in listdir(test_folder) if splitext(f)[1] == extension]
                
                if params['balanced'] and index != min_train_class:
                    train_files = train_files[0:min_num_train]
                if params['balanced'] and index != min_test_class:
                    test_files = test_files[0:min_num_test]
                
                random.shuffle(train_files)
                random.shuffle(test_files)
                
                num_train = int(len(train_files) * train_percent)
                num_valid = int(len(train_files) * valid_percent)
                num_test = len(test_files)
                
                file_storage['train'] += train_files[:num_train]
                label_storage['train'] += [label] * num_train
                file_storage['valid'] += train_files[num_train:num_train+num_valid]
                label_storage['valid'] += [label] * num_valid
                file_storage['test'] += test_files
                label_storage['test'] += [label] * num_test       
                
            if verbose:
                print("Getting samples for class " + str(index)+"...")
                print("There are " + str(num_train) + " training, " + str(num_valid) + "validation, " +
                      str(num_test) + " test samples.")
                
        
        #   make file_storage and label_storage into attributes
        self.file_storage = file_storage
        self.label_storage = label_storage
        
        #   zip and shuffle data -- stype is set type
        for stype in file_storage:
            loc_dataset[stype] = list(zip(file_storage[stype], label_storage[stype]))       #   contains tuples of (file_loc, label)
            random.shuffle(loc_dataset[stype])            
        
        if verbose is True and self.superspeedmode is True:
            print("Superspeedmode is activated.  Computing and saving arrays for all images...") 
        
        #   create batches 
        for stype in loc_dataset:
            self.X_batches[stype] = []
            self.y_batches[stype] = []
            for x in range(0, len(loc_dataset[stype]), batch_size):
                one_batch = loc_dataset[stype][x:x+batch_size]
                img_batch = []
                label_batch = []
                for sample_idx in range(0, len(one_batch)):     #   for each file in a batch
                    if self.superspeedmode is True:             #    computes np.array for all samples and save in memory   
                        img_batch.append(get_image_array(one_batch[sample_idx][0], self.grayscale, self.target_size))      #   convert location to np.array
                    else:
                        img_batch.append(one_batch[sample_idx][0])
                    label_batch.append(one_batch[sample_idx][1])
                self.X_batches[stype].append(img_batch)
                self.y_batches[stype].append(label_batch)
        
                #   self.X_batches[stype] contains the batches for each stype
                #   type is [batch[string, ], ]
        
            self.num_batches[stype] = math.ceil(len(loc_dataset[stype])/batch_size)     #   calculate the number of batches
            
        if verbose is True:
            print("There are", len(loc_dataset['train']), "training samples and", len(self.y_batches['train']), "training batches.")
            print("There are", len(loc_dataset['valid']), "validation samples and", len(self.y_batches['valid']), "validation batches.")
            print("There are", len(loc_dataset['test']), "test samples and", len(self.y_batches['test']), "test batches.")
        
        del loc_dataset, file_storage, label_storage, one_batch, img_batch, label_batch
        
    def get_dataset_names_and_labels(self, stype):
        """
        Returns the filenames and labels of a given dataset type (stype) in its output order.  Batches are broken
        down.
        Inputs:
            stype (str) :                       dataset type ('train', 'valid', 'test')
        Returns:
            names (list[str, ]):        list of filenames
            labels (list[int, ]):       list of labels corresponding to 
                                                    filenames
        """
        if stype not in ['train', 'valid', 'test']:
            raise ValueError("stype not train, valid, or test")
        
        names = []
        labels = []
        names = [filename for batch in self.X_batches[stype] for filename in batch]
        labels = [label for batch in self.y_batches[stype] for label in batch]
        
        return names, labels
    
    
    def load_batch(self, stype, batch_idx=None):
        """
        Loads a single batch of set stype.  If batch_idx is specified, loads
        batch of set stype at index batch_idx.
        Note this function does not loop or stop when no batches, so use get_num_batches() 
        to ensure that it does not overflow.
        Arguments:
            stype (str) :               set type ("train", "valid", "test")
            batch_idx (int) :           index of batch to load
        Returns:
            X_batch (list[np.array]) :     a single batch of stype as a list.  Each sample is
                                                 a np.array(image)
            y_batch (list[int/str]) :      a single batch of stype as a list.  Each sample is
                                                a label (str) or class (int)
        """
        
        self.batch_counters = {'train': 0, 'valid': 0, 'test': 0}  #   counter for each batch
        
        if batch_idx is None:
            loc_batch = self.X_batches[stype][self.batch_counters[stype]]
            label_batch = self.y_batches[stype][self.batch_counters[stype]]
            self.batch_counters[stype] += 1         #   increment to next batch
        else:
            loc_batch = self.X_batches[stype][batch_idx]
            label_batch = self.y_batches[stype][batch_idx]
        
        if self.superspeedmode is True:
            X_batch = loc_batch
        else:
            #   create np.array image batch
            X_batch = []
            for idx in range(0, len(loc_batch)):
                X_batch.append(get_image_array(loc_batch[idx], self.grayscale, self.target_size))
        
        y_batch = label_batch
        
        return X_batch, y_batch
        
    def restart(self, stype=""):
        """
        Restart the batch order of the dataloader.  If stype is not specified, restarts
        for all set types.
        Arguments:
            stype (str) :       set type to restart counter ("train", "valid", "test")
        """
        if stype == "":
            for stype in self.batch_counters:
                self.batch_counters[stype] = 0
        else:
            self.batch_counters[stype] = 0
    
    def get_num_batches(self, stype):
        """
        Get the number of batches in each set.
        Arguments:
            stype (str) :       set type ("train", "valid", "test")
        Returns:
            int :               number of batches in stype
        """
        return self.num_batches[stype]
    
    def get_dataset_images_and_labels(self, stype=""):
        """
        Get dataset of set stype or entire dataset if stype is not specified.
        (Note this function is not prioritized for efficiency, compared to load_batch().
        That is, full computation runs everytime you call it.)
        Arguments:
            stype (str) :               set type ("train", "valid", "test")
        Returns:
            X_dataset (list/dict[np.array]) :     a dataset of set stype as a dict/list.  Each sample is 
                                                    a np.array(image)
            y_dataset (list/dict[int/str]) :      a dataset of set stype as a dict/list.  Each sample is
                                                    a label (str) or a class (int)
        """
        if stype == "":
            loc_dataset = {}
            label_dataset = {}
            for stype in self.batches:
                #   loc_dataset is a dictionary of lists
                loc_dataset[stype] = list(itertools.chain.from_iterable(self.X_batches[stype]))   #   flattens 2d list to 1d list
                label_dataset[stype] = list(itertools.chain.from_iterable(self.y_batches[stype]))
        else:
            #   loc_dataset is a list
            loc_dataset = list(itertools.chain.from_iterable(self.X_batches[stype]))
            label_dataset = list(itertools.chain.from_iterable(self.y_batches[stype]))
        
        if self.superspeedmode is True:
            X_dataset = loc_dataset           #   dataset is already in np.array
        else:
            if stype == "":
                X_dataset = {}
                for stype in loc_dataset:
                    X_dataset[stype] = []
                    for sample_idx in range(0, len(loc_dataset[stype])):
                        X_dataset[stype].append(get_image_array(loc_dataset[stype][sample_idx], self.grayscale, self.target_size))
            else:
                X_dataset = []
                for sample_idx in range(0, len(loc_dataset)):       
                    X_dataset.append(get_image_array(loc_dataset[sample_idx], self.grayscale, self.target_size))
        
        y_dataset = label_dataset
        
        return X_dataset, y_dataset
    
    def print_dataset_files(self, stype="", save_dir=""):
        """
        Prints the filenames making up a dataset into a .txt file.  Note that the order 
        of the files are not in the output (randomized) order, but their original file order.
        Arguments:
            stype (str) :               set type ("train", "valid", "test")
            save_dir (str) :            directory to output .txt files to
        """
        
        if stype == "":
            stypes = ["train", "valid", "test"]
        else:
            stypes = [stype]
        
        if save_dir != "":
            if save_dir[-1] != "/":
                save_dir += "/"
        
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        
        for stype in stypes:
            if len(self.file_storage[stype]) != len(self.label_storage[stype]):
                raise Exception("len(self.file_storage[stype]) = " + str(len(self.file_storage[stype])) + \
                                "    len(self.label_storage[stype]) = " + str(len(self.label_storage[stype])))
            print(stype)
            print("Number of files: ", len(self.file_storage[stype]))
            print("Number of labels: ", len(self.label_storage[stype]))
            f = open(save_dir+stype+".txt", "w+")
            for idx in range(0, len(self.file_storage[stype])):
                write_string = str(self.file_storage[stype][idx]) + " " + str(self.label_storage[stype][idx])
                #print(write_string)
                f.write(write_string+"\n")
            f.close()
    

def get_image_array(file, grayscale=False, target_size=None):
    """
    Converts an image to a np.array.
    
    Arguments:
        file (str) :                        location and filename of image
        grayscale (Boolean) :               whether image is grayscale or not
        target_size (tuple(int,int)) :      output np.array size 
    Returns:
        np.array :                  a np.array of the image
    """
    img = image.load_img(file, grayscale=grayscale, target_size=target_size)
    return image.img_to_array(img)


def plot_image(image):
    """
    Plot np.array as an image.

    """
    plt.imshow(image)

def load_from_npy(X_file, y_file):
    """
    Load MNIST-like data from npy files as an np.arrays
    Arguments:
        X_file (str) :              .npy file location of training features
        y_file (str) :              .npy file location of training labels
    Returns:
        Numpy arrays representing the features and labels
            - shape (num_samples, height, width)
            - shape (num_samples, )
    """
    X = np.load(X_file)
    y = np.load(y_file)
    
    print(X.shape)
    print(y.shape)
    
    return X, y

def save_npy_as_images(X_file, y_file, save_dir=""):
    """
    Saves images stored in .npy files to images.
    """
    from PIL import Image
    
    X, y = load_from_npy(X_file, y_file)
    
    if X.shape[0] != y.shape[0]:
        raise Exception("Dimension mismatch for X and y.")
    
    #   create directory for categorization
    max_category = np.max(y)
    min_category = np.min(y)
    print("Max category: ", max_category, "    Min category: ", min_category)
    
    if save_dir != "":
        if save_dir[-1] != "/":
            save_dir += "/"
            
    for cat in range(min_category, max_category+1):
        if os.path.exists(save_dir+str(cat)+"/") is False:
            os.makedirs(save_dir+str(cat)+"/")
    
    #   for every image in every label, save as .png
    for idx in range(0, y.shape[0]):
        label = y[idx]
        image = X[idx]
        print("Saving image ", idx, "...")
        im =Image.fromarray(image)
        im.save(save_dir+str(label)+"/"+str(idx)+".png")

if __name__ == "__main__":
    
    """
    params = {
            'labels':           ['tabby', 'siamese'],
            'label_type':       'int',
            'file_locs':        ["../data/tabby_cat", "../data/siamese_cat"],
            'file_exten':       '.JPEG',
            'target_size':      (224, 224),
            'set_ratio':        [0.8, 0.1, 0.1],
            'batch_size':       3,
            'superspeedmode':   False                #   trades off memory efficiency for less computation (USE AT YOUR OWN RISK)
    }
    
    dataloader = Dataloader(params, rseed=0)
    valid_batch = dataloader.load_batch('valid')
    #print(valid_batch)
    X_valid_batch, y_valid_batch = dataloader.load_batch('valid')
    print("Display image...")
    plot_image(X_valid_batch[0])
    print("Continue.")
    print(X_valid_batch[0].shape)
    print(X_valid_batch[1].shape)
    dataloader.restart()
    X_valid_batch, y_valid_batch = dataloader.load_batch('valid')
    print(X_valid_batch[0].shape)
    print(X_valid_batch[1].shape)
    print(dataloader.get_num_batches("valid"))
    X_dataset, y_dataset = dataloader.get_dataset("valid")
    print(len(y_dataset))
    #   delete dataloader after you're done using if using superspeedmode (clears memory)
    del dataloader
    """