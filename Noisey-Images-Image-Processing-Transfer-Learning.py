#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:14:14 2023

@author: idu
"""
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt
from PIL import Image
from termcolor import colored  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img



########################### Adding Noise to The Original Images in the validation set [Gaussian and Salt-and-Pepper Noise] ########################
###########################^^############################^^^^################################################^^^^^^^^^

######### Function to add Gaussian noise to an image#########
def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image"""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are valid
    return noisy_image

## For Dİfferent Sigma level - Choose one and replace the one above with
# 2nd sigma = 5
def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image"""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are valid
    return noisy_image

# 3rd sigma = 15
def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image"""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are valid
    return noisy_image


# 4th sigma = 35
def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image"""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are valid
    return noisy_image


########## Function to add salt-and-pepper noise to an image #########
def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    """Add salt and pepper noise to an image"""
    noisy_image = np.copy(image)
    total_pixels = image.size
    
    # Add salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords] = 1

    # Add pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords] = 0

    return noisy_image

########## Function to load an image#########
def load_image(image_path, target_size=(224, 224)):
    """Load a grayscale image"""
    img = load_img(image_path, target_size=target_size, color_mode="grayscale")
    img_array = img_to_array(img)
    return img_array

########## Function to save an image#########
def save_image(image, save_path):
    """Save a grayscale image to the specified path in JPEG format"""
    save_img(save_path, image, data_format='channels_last', file_format='jpeg')

# Path to validation set folder

#gaussian_output_path is used for saving images with Gaussian noise.
#salt_and_pepper_output_path is used for saving images with salt-and-pepper noise.
#The images with Gaussian noise will be saved in the val-gaussian-noise-added directory, and the images with salt-and-pepper noise will be saved in the val-salt-and-pepper-noise-added directory.

validation_set_path = '/home/idu/Desktop/COV19D/val'
gaussian_output_path = '/home/idu/Desktop/COV19D/val-gaussian-noise-added'
salt_and_pepper_output_path = '/home/idu/Desktop/COV19D/val-salt-and-pepper-noise-added'

# Function to add noise to images and save them
def add_noise_and_save_images(image_paths, output_path, validation_set_path, noise_type='gaussian', **kwargs):
    for image_path in image_paths:
        try:
            # Determine output path for noisy image
            relative_path = os.path.relpath(image_path, start=validation_set_path)
            noisy_image_save_path = os.path.join(output_path, relative_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(noisy_image_save_path), exist_ok=True)
            
            # Load image
            image = load_image(image_path)
            
            # Add noise
            if noise_type == 'gaussian':
                noisy_image = add_gaussian_noise(image, **kwargs)
            elif noise_type == 'salt_and_pepper':
                noisy_image = add_salt_and_pepper_noise(image, **kwargs)
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
            
            # Save noisy image
            save_image(noisy_image, noisy_image_save_path)
            print(f"Saved noisy image: {noisy_image_save_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

# Get all image paths recursively from covid and non-covid directories
covid_image_paths = []
non_covid_image_paths = []

for root, dirs, files in os.walk(os.path.join(validation_set_path, 'covid')):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            covid_image_paths.append(os.path.join(root, file))

for root, dirs, files in os.walk(os.path.join(validation_set_path, 'non-covid')):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            non_covid_image_paths.append(os.path.join(root, file))

# Process and save noisy covid images with Gaussian noise
print("Processing and saving COVID images with Gaussian noise:")
add_noise_and_save_images(covid_image_paths, gaussian_output_path, validation_set_path, noise_type='gaussian', sigma=25)

# Process and save noisy non-covid images with Gaussian noise
print("\nProcessing and saving Non-COVID images with Gaussian noise:")
add_noise_and_save_images(non_covid_image_paths, gaussian_output_path, validation_set_path, noise_type='gaussian', sigma=25)

# Process and save noisy covid images with salt-and-pepper noise
print("\nProcessing and saving COVID images with salt-and-pepper noise:")
add_noise_and_save_images(covid_image_paths, salt_and_pepper_output_path, validation_set_path, noise_type='salt_and_pepper', salt_prob=0.05, pepper_prob=0.05)

# Process and save noisy non-covid images with salt-and-pepper noise
print("\nProcessing and saving Non-COVID images with salt-and-pepper noise:")
add_noise_and_save_images(non_covid_image_paths, salt_and_pepper_output_path, validation_set_path, noise_type='salt_and_pepper', salt_prob=0.05, pepper_prob=0.05)


#######################################################################################
######################## Processing Newly Created Noisey IMages ############################################
#######################################################################################


### Slices Deletion
#########################
import os
import cv2
import re

# Define path of images to be processed [COV19-CT-DB]
val_dir = '/home/idu/Desktop/COV19D/val-noise-added/covid'  # repeat for "non-covid" folder


main_dir = val_dir ## Change this directory as needed to do slices deletion in

# Define the percentage of images to delete
percentage_to_delete = 40  # Adjust this value as needed

# Function to calculate the number of images to delete
def calculate_images_to_delete(total_count):
    images_to_delete = int((percentage_to_delete / 100) * total_count)
    return images_to_delete

# Function to extract the image number from the filename
def extract_image_number(filename):
    match = re.match(r"(\d+).jpg", filename)
    if match:
        return int(match.group(1))
    return float('inf')  # Use a large value for files that don't match the pattern

# Process each subfolder in the main directory

for subfolder in os.listdir(main_dir):
    subfolder_path = os.path.join(main_dir, subfolder)
    
    if os.path.isdir(subfolder_path):
        # List all files in the subfolder
        files = os.listdir(subfolder_path)
        files.sort(key=lambda x: extract_image_number(x))  # Sort files by image number with handling None

        total_count = len(files)
        
        if total_count > 1:
            images_to_delete = calculate_images_to_delete(total_count)
            
            print(f"Processing subfolder: {subfolder}")
            
            # Print the list of files before deletion
            print("Files before deletion:", files)
            
            # Delete a percentage of images, keeping centered ones
            for i in range(images_to_delete):
                # Delete images at the beginning and end
                file_to_delete_first = os.path.join(subfolder_path, files[i])
                file_to_delete_last = os.path.join(subfolder_path, files[-(i + 1)])
                
                try:
                    print(f"Deleting image: {file_to_delete_first}")
                    os.remove(file_to_delete_first)
                except FileNotFoundError:
                    print(f"File not found: {file_to_delete_first}")

                try:
                    print(f"Deleting image: {file_to_delete_last}")
                    os.remove(file_to_delete_last)
                except FileNotFoundError:
                    print(f"File not found: {file_to_delete_last}")
            
            # Print the list of files after deletion
            #files_after_deletion = os.listdir(subfolder_path)
            #print("Files after deletion:", files_after_deletion)

print("Deletion process completed.")



### Slices Cropping of the Newly Created Noisey Images
####################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#path for images to be processed
folder_path = val_dir ## Change this to the directory to do the slices cropping in

# Specify the new size and cropping position
new_height = 227
new_width = 300
crop_x = 99
crop_y = 160

for sub_folder in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, sub_folder)
    
    print(f'Processing subfolder: {sub_folder}')
    
    for file_name in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, file_name)
        
        # Check if the file is an image (you can add more image extensions if needed)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            #print(f'Processing file: {file_name}')
            
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
            
            # Check if the image was loaded successfully
            if img is not None:
                # Crop the image
                img_cropped = img[crop_y:crop_y+new_height, crop_x:crop_x+new_width]
                
                # Save the cropped image by overwriting the original image
                cv2.imwrite(file_path, img_cropped)
                
                #print(f'Cropped and saved: {file_name}')
            else:
                print(f'Failed to load image: {file_name}')

print('finished')


####################################################################################
######################## Transfer Learning Models for classification of Noisey Images 
####################################################################################

# Loaing our saved model
SIZE =224

model = keras.models.load_model("/home/idu/Desktop/COV19D/saved-models/CNN model/imageprocess-sliceremove-cnn.h5")


#Making predictions on the validation set of noisey images COV19-CT-DB

## Choosing the directory where the test/validation data is at
from termcolor import colored  # Importing colored for colored console output
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

folder_path = '/home/idu/Desktop/COV19D/val-noise-added/covid'  # Change as needed

extensions0 = []
extensions1 = []
extensions2 = []
extensions3 = []
extensions4 = []
extensions5 = []
extensions6 = []
extensions7 = []
extensions8 = []
extensions9 = []
extensions10 = []
extensions11 = []
extensions12 = []
extensions13 = []
covidd = []
noncovidd = []
coviddd = []
noncoviddd = []
covidddd = []
noncovidddd = []
coviddddd = []
noncoviddddd = []
covidd6 = []
noncovidd6 = []
covidd7 = []
noncovidd7 = []
covidd8 = []
noncovidd8 = []

results = 1
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)

        try:
            c = load_img(file_path, color_mode='grayscale', target_size=(227, 300))
            c = img_to_array(c)
            c = np.expand_dims(c, axis=0)
            c /= 255.0
            result = model.predict(c)  # Probability of 1 (non-covid)

            if result > 0.97:  # Class probability threshold is 0.97
                extensions1.append(results)
            else:
                extensions0.append(results)
            if result > 0.90:  # Class probability threshold is 0.90
                extensions3.append(results)
            else:
                extensions2.append(results)
            if result > 0.70:  # Class probability threshold is 0.70
                extensions5.append(results)
            else:
                extensions4.append(results)
            if result > 0.40:  # Class probability threshold is 0.40
                extensions7.append(results)
            else:
                extensions6.append(results)
            if result > 0.50:  # Class probability threshold is 0.50
                extensions9.append(results)
            else:
                extensions8.append(results)
            if result > 0.15:  # Class probability threshold is 0.15
                extensions11.append(results)
            else:
                extensions10.append(results)
            if result > 0.05:  # Class probability threshold is 0.05
                extensions13.append(results)
            else:
                extensions12.append(results)
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            continue

    # The majority voting at Patient's level
    if len(extensions1) > len(extensions0):
        print(fldr, colored("NON-COVID", 'red'), len(extensions1), "to", len(extensions0))
        noncovidd.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions0), "to", len(extensions1))
        covidd.append(fldr)
    if len(extensions3) > len(extensions2):
        print(fldr, colored("NON-COVID", 'red'), len(extensions3), "to", len(extensions2))
        noncoviddd.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions2), "to", len(extensions3))
        coviddd.append(fldr)
    if len(extensions5) > len(extensions4):
        print(fldr, colored("NON-COVID", 'red'), len(extensions5), "to", len(extensions4))
        noncovidddd.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions5), "to", len(extensions4))
        covidddd.append(fldr)
    if len(extensions7) > len(extensions6):
        print(fldr, colored("NON-COVID", 'red'), len(extensions7), "to", len(extensions6))
        noncoviddddd.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions6), "to", len(extensions7))
        coviddddd.append(fldr)
    if len(extensions9) > len(extensions8):
        print(fldr, colored("NON-COVID", 'red'), len(extensions9), "to", len(extensions8))
        noncovidd6.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions8), "to", len(extensions9))
        covidd6.append(fldr)
    if len(extensions11) > len(extensions10):
        print(fldr, colored("NON-COVID", 'red'), len(extensions11), "to", len(extensions10))
        noncovidd7.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions10), "to", len(extensions11))
        covidd7.append(fldr)
    if len(extensions13) > len(extensions12):
        print(fldr, colored("NON-COVID", 'red'), len(extensions13), "to", len(extensions12))
        noncovidd8.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions12), "to", len(extensions13))
        covidd8.append(fldr)

    extensions0 = []
    extensions1 = []
    extensions2 = []
    extensions3 = []
    extensions4 = []
    extensions5 = []
    extensions6 = []
    extensions7 = []
    extensions8 = []
    extensions9 = []
    extensions10 = []
    extensions11 = []
    extensions12 = []
    extensions13 = []

# Checking the results
# print(len(covidd))
# print(len(coviddd))
print(len(covidddd))
print(len(coviddddd))
print(len(covidd6))
print(len(covidd7))
# print(len(covidd8))
# print(len(noncovidd))
# print(len(noncoviddd))
print(len(noncovidddd))
print(len(noncoviddddd))
print(len(noncovidd6))
print(len(noncovidd7))
# print(len(noncovidd8))
# print(len(covidd+noncovidd))
# print(len(coviddd+noncoviddd))
print(len(covidddd+noncovidddd))
print(len(coviddddd+noncoviddddd))
print(len(covidd6+noncovidd6))
print(len(covidd7+noncovidd7))
# print(len(covidd8+noncovidd8))