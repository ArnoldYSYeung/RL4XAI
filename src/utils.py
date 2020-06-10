"""
Utility functions.
Date :                      August 23rd, 2019

Updates:
    2019-08-23                      Add normalize_array
    2019-09-12                      Add plot_numpy_images()
    2019-09-20                      Make plot_numpy_images() compatible with grayscale images
    2019-09-21                      Modify plot_numpy_images() to include label in filename
    2019-10-04                      Fixed normalize_array() bug where minmax are the same
    2019-11-05                      Add print_iterables_file()
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
import textwrap

def normalize_array(img, minmax=[0, 255], scale_range=[-1, 1], dtype=float):
    """
    Scales the values of an image array to between -1 to 1.
    Arguments:
        img (np.array) :                    image array to scale
        minmax (list[int, int]) :           list containing the minimum and maximum values of img
        scale_range (list[int, int]) :      list containing the range to scale to
        dtype(datatype) :                   data type of values of returned image array
    Returns:
        img (np.array) :                    array with normalized values
    """
    
    if len(minmax) != 2 or len(scale_range) != 2:
        raise ValueError("Input lists are not of length 2.")
    
    min_value, max_value = minmax
    min_scale, max_scale = scale_range
    
    if min_value == max_value:
        print("WARNING: Min value and max value are the same.")
        img = np.ones(img.shape) * (max_scale-min_scale)
        img = img.astype(dtype)
        return img
        
        
    if min_scale == max_scale:
        raise ValueError("Scale does not exist.")
    
    normalizer = (max_value-min_value)/(max_scale-min_scale)
    img = (img-min_value)/normalizer + min_scale
    img = img.astype(dtype)             #   change the data type of the array
    return img


def add_caption_to_image(image, caption, scale=1, caption_height=10, font_file=""):
    """
    Add a caption to an image.
    Arguments:
        image (np.array) :          numpy array of an image (height, width, layers)
        caption (str) :             text to add as caption
        scale (int) :               how much to scale the image (to reduce low res situations)
        caption_height (int) :      height of caption
        font_file (str) :           location of font file for caption
    Returns:
        np.array of rescaled image with added caption (height, width, layers)
    """
    
    offset = (0, 0)
    height, width, _ = image.shape
    
    background_size = (scale*width, scale*(height+caption_height))
    background = Image.new(mode="RGB", size=background_size, color=(255,255,255))     #   background color white
    draw = ImageDraw.Draw(background)
    
    #   create caption
    text_size = 10*scale
    if font_file == "":
        font = ImageFont.load_default()         #   default font
    else:
        font = ImageFont.truetype(font=font_file, size=text_size)
    
    #   add caption to image
    print("Min Max: ", np.min(image), np.max(image))
    
    img = Image.fromarray(np.uint8(image*255)).convert("RGB")    #   greyscale
    img = img.resize((width*scale, height*scale))     #   resize the inputted image
    width, height = img.size
    background.paste(img, box=offset)
    offset = (offset[0], offset[1]+height)
    
    #   add caption to background
    draw.text(offset, caption, (0, 0, 0), align="center", font=font)    
    text_width, text_height = draw.textsize(caption, font=font)
    
    return np.array(background.convert("RGB"))


def plot_numpy_images(images_and_labels, filename="", indices=[], concat=False, 
                      mark_label=False, font_file="", label_names=[], save_dir=""):
    """
    Plot numpy arrays as images.  Each image in the numpy array will be plotted as an
    individual image.
    Arguments:
        images_and_labels (tuple(np.array, )) :
            images (np.array) :             array representing images of shape (num_samples * height * width * num_layers)
            labels (np.array) :             array containing the labels of each image (num_samples, )
        filename (str) :                filename to save image as (not saved if "")
        indices (list[int, ]) :          list of indices to append at the end of filename to denote
                                            multiple images (if [], use range from 0 to num_samples)
        concat (Boolean) :              whether to concatenate multiple images
        mark_label (Boolean) :          whether to add label text underneath the image
        font_file (str) :               file of font used for label text (only if mark_label=True)
        label_names (lst[str, ]) :      names of each label class
        save_dir (str) :                directory to save images in (default directory if "")
    """
    
    images, labels = images_and_labels
    border_width = 2            #   number of pixels separating each prototype image
    
    #   get shape
    num_samples, height, width, num_layers = images.shape
    
    if save_dir != "" and save_dir[-1] != "/":
            save_dir += "/"
    
    if num_samples <= 1:
        plt.imshow(images[0, :, :, :])
        if filename != "":
            save_name = save_dir+filename+"_" + str(labels[0]) + ".png"
            plt.savefig(save_name)
        plt.show()
    else:
        if indices == []:
            indices = [n for n in range(0, num_samples)]
        
        if len(indices) != num_samples:
            raise ValueError("Number of indices do not match number of images.")
        
        #   plot and save every image
        if concat:
            concat_images = None
        
        caption_height = 10
        scale = 5
        for idx in range(0, num_samples):
            image = images[idx, :, :, :]
            
            if mark_label:
                image = add_caption_to_image(image, str(label_names[labels[idx]]), scale=scale, 
                                             caption_height=caption_height, font_file=font_file)
                save_name = save_dir+filename+"_"+str(indices[idx])+ "_" + str(labels[idx]) + ".png"
                plt.imshow(image)
                plt.savefig(save_name)
                
            if image.shape[-1] == 1:            #   if grayscale
                image = image.squeeze(axis=-1)
            if concat and idx == 0:
                concat_images = image
            elif concat:
                if mark_label:
                    border = np.ones((scale*(height+caption_height), scale*border_width, num_layers),
                                     dtype=np.uint8).squeeze()*255
                else:
                    border = np.ones((height, border_width, num_layers), 
                                     dtype=np.uint8).squeeze() * 255        #   white border
                #   concatenate along width
                #print("Border: ", border.shape, border.dtype)
                concat_images = np.concatenate((concat_images, border), axis=1)
                concat_images = np.concatenate((concat_images, image), axis=1)
                #print("Concat: ", concat_images.shape, concat_images.dtype)
                #plt.imshow(concat_images)
                #plt.savefig(save_dir+"concat"+str(idx)+".png")
            if concat is False:
                plt.imshow(image)
                if filename != "":
                    save_name = save_dir+filename+"_"+str(indices[idx])+ "_" + str(labels[idx]) + ".png"
                    plt.savefig(save_name)
        
        if concat:
            plt.imshow(concat_images)
            if filename != "":
                indice_string = [str(i) for i in indices[0:num_samples]]
                indice_string = "_".join(indice_string)
                label_string = [str(label) for label in labels[0:num_samples]]
                label_string = "_".join(label_string)
                save_name = save_dir+filename+"_"+ indice_string + "_" + label_string + ".png"
                plt.savefig(save_name)
                    
    del images
    return save_name

def print_iterables_file(iterables, save_name, save_dir="", separator=" "):
    """
    Print lists into a .txt file.
    Arguments:
        iterables (list[list[], ]) :            list of iterables of the same length
        save_name (str) :                       filename
        save_dir (str) :                        directory to save file in
        separator (str) :                       what to use for separating values of different iterables
    """
    
    length = len(iterables[0])
    for iterable in iterables:
        if len(iterable) != length:
            raise IndexError("Lengths of iterables are different.")
    
    if save_dir != "":
        if save_dir[-1] != "/":
            save_dir += "/"
        
    if save_dir != "" and os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    f = open(save_dir+save_name+".txt", "w+")
    
    for idx in range(0, length):
       write_string = ""
       for iterable in iterables:
           write_string += str(iterable[idx]) + separator
       f.write(write_string+"\n")
    f.close()

def convert_npz_to_greyscale_png(npz_image_file, npz_label_file, save_dir, 
                                 black_val=0, white_val=255):
    """
    Convert and save greyscale images compressed in .npz file to .png files.
    Arguments:
        npz_image_file (str) :              directory of .npz file containing images
        npz_label_file (str) :              directory of .npz file containing labels
        save_dir (str) :                    directory to save .png files in
        black_val (int) :                   value of pixel to map to darkest shade
        white_val (int) :                   value of pixel to map to lightest shade
    """
    
    #   format save_dir
    if save_dir[-1] != "/":
        save_dir += "/"
    
    #   load image array
    images = np.load(npz_image_file)
    images_files = images.files
    images = images[images_files[0]]
    print("Image shape: ", images.shape)
    
    #   load label array
    labels = np.load(npz_label_file)
    labels_files = labels.files
    labels = labels[labels_files[0]]
    print("Label shape: ", labels.shape)
    
    existing_labels = []
    import scipy.misc
    
    for i in range(0, images.shape[0]):
        if i % 100 == 0:
            print(str(i)+" ", end="")
        
        label = labels[i]
        image = images[i]
        if label not in existing_labels:
            directory = save_dir + str(label)
            if os.path.isdir(directory) is False:
                os.mkdir(directory)
            existing_labels.append(label)
        #   save image
        scipy.misc.toimage(image, cmin=black_val, cmax=white_val).save(save_dir
                          +str(label)+"/"+str(i)+".png")
        
if __name__ == "__main__":

    pass