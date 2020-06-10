#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combining multiple images and text (captions) to create a full image.
Date :                  October 8th, 2019

"""

import PIL
from PIL import Image, ImageDraw, ImageFont
import textwrap

def get_background_size(text, image_series_lst, border, text_size):
    """
    Calculates the suitable background size 
    """
    
    #   get combined height of all images and max image_series width
    image_heights = 0
    max_width = 0
    max_num_images = 0
    for image_series in image_series_lst:
        num_images = len(image_series)
        series_width = 0 
        max_height = 0
        for file in image_series:
            image = Image.open(file)
            width, height = image.size
            if height > max_height:
                max_height = height
            series_width += width
        image_heights += max_height
        if series_width > max_width:
            max_width = series_width
        if max_num_images < num_images:
            max_num_images = num_images
    
    background_height = len(text) * text_size + image_heights + border[1]*len(image_series_lst)
    background_width = max_width + border[0] * max_num_images 
    
    return (background_width, background_height)

    #   get max text width 

def create_explanation_images(text, image_series_lst, image_params, save_dir="", save_name="image.png", 
                              background_size=None, charwidth=90):
    """
    Combine images to create explanations with text.  Each line of text will be 
    followed by a series of images.
    Arguments:
        text (list[]) :                         list of text to display
        image_series_lst(list[list[], ]) :      list of image_series
        image_params (dict):
            border (tuple(int, int)) :              border between each image (width, height)
            font_file (str) :                       location of file storing font to use for text
            text_size (int) :                       size of text
        save_dir (str) :                        directory to save final image to
        background_size (tuple(int, int)) :     size of the output image (pixels)
        charwidth (int) :                       how many characters (not pixels) per line for 
                                                    explanation_text
    """
    
    text_size = image_params['text_size']
    font_file = image_params['font_file']
    border = image_params['border']
    
    #   check if there is an image series for every text
    if len(text) < len(image_series_lst):
        print("WARNING: Less text than image_series_lst.  There are ", len(text),
              " texts and ", len(image_series_lst), " images.")
    elif len(text) > len(image_series_lst):
        print("WARNING: More text than image_series_lst.  There are ", len(text),
              " texts and ", len(image_series_lst), " images.")
        
    num_lines = max(len(text), len(image_series_lst))
    
    if background_size is None:
        #   calculate expected size (kinda works....)
        background_size =get_background_size(text, image_series_lst, border, text_size)
    print("Using background_size of : ", background_size)
    
    background = Image.new(mode="RGBA", size=background_size)
    draw = ImageDraw.Draw(background)
    
    offset = (0, 0)
    
    for idx in range(0, num_lines):
        if idx < len(text):
            if font_file == "":
                font = ImageFont.load_default()         #   default font
            else:
                font = ImageFont.truetype(font=font_file, size=text_size)
            caption = text[idx]
            #   wrap caption into multiple lines
            caption_lines = textwrap.wrap(caption, width=charwidth)
            #   draw multiple lines of caption
            for line in caption_lines:
                #print("Text Offset: ", offset)
                draw.text(offset, line, (0, 0, 0), align="center", font=font)    
                text_width, text_height = draw.textsize(caption, font=font)
                offset = (0, offset[1]+border[1]+text_height)
            
        if idx < len(image_series_lst):
            max_height = 0
            for file in image_series_lst[idx]:
                #print("Image Offset: ", offset)
                img = Image.open(file)
                width, height = img.size
                if height > max_height:
                    max_height = height
                background.paste(img, box=offset)
                #   update offset for next image
                offset = (offset[0]+width+border[0], offset[1])    
            offset = (0, offset[1]+max_height+border[1])
    
    if save_dir != "" and save_dir[-1] != "/":
        save_dir += "/"
    
    background.save(save_dir + save_name)


if __name__ == "__main__":
    
    save_dir = "./"
    text = ['hello it is nice to meet you', 'morning']
    image_series_lst = [[save_dir+'lime_0.png', save_dir+'lime_1.png'], [save_dir+'lime_2.png', save_dir+'lime_3.png']]
    
    image_params = {'border' :          (0, 0),
                    'font_file' :       'Arial.ttf',
                    'text_size' :       15
            }
    
    create_explanation_images(text, image_series_lst, image_params, save_dir=save_dir, save_name="explanations.png")