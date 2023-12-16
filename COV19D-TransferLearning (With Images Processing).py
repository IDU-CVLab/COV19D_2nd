#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 15:50:19 2021

@author: idu
"""
#######################################################################################
######################## Images Processing ############################################
#######################################################################################


### Slices Deletion
#########################
import os
import cv2
import re

# Define path of images to be processed [COV19-CT-DB]
train_dir = '/home/idu/Desktop/COV19D/train-processed/covid/'
val_dir = '/home/idu/Desktop/COV19D/val-processed/non-covid/'
test_dir = '/home/idu/Desktop/COV19D/ICASSP-test/11' ## ECCV COV19-CT-DB

main_dir = test_dir ## Change this directory as needed to do slices deletion in

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



### Slices Cropping
###################

#path for images to be processed
folder_path = test_dir ## Change this to the directory to do the slices cropping in

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
######################## Transfer Learning Models for classification################
####################################################################################

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow import keras
from keras.models import load_model
from keras import backend as K
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import Image
from termcolor import colored

#Using image datagenerator for Generating data with rescaling and binary labels from the images (rgb images)
batch_size = 32
SIZE = 224  ## Resizing images to 224x224

train_datagen = ImageDataGenerator(rescale=1./255)
                                   
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train-processed/', 
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        color_mode='rgb', 
        class_mode='binary')  

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/val-processed/',  
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        color_mode='rgb',
        class_mode='binary')

### Using pretrained Xception model
Model_Xcep = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
#Model_VGG = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))

for layer in Model_Xcep.layers:
	layer.trainable = False
    
#for layer in Model_VGG.layers:
#	layer.trainable = False
    
Model_Xcep.summary()

###### modified the output

model = tf.keras.Sequential([
    Model_Xcep, 
    tf.keras.layers.GlobalAveragePooling2D(), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#model = tf.keras.Sequential([
#    Model_VGG, 
#    tf.keras.layers.GlobalAveragePooling2D(), 
#    tf.keras.layers.Dense(128, activation='relu'),
#    tf.keras.layers.BatchNormalization(), 
#    tf.keras.layers.Dropout(0.2), 
#    tf.keras.layers.Dense(1, activation='sigmoid')
#])
    
    
#model = tf.keras.models.load_model ("Modified_Xception1.h5")    
model.summary()  

# Adding callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("/home/idu/Desktop/COV19D/saved-models/Transfer Learning/imageprocessed-Xception.h5", save_best_only=True, verbose = 0),
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_accuracy', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# Compiling the model
model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001),
              loss = 'binary_crossentropy',
              metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])

#### TRainging the model
history = model.fit(train_generator, 
                    validation_data=val_generator, epochs=15, 
                    callbacks=[callbacks])

model.save("/home/idu/Desktop/COV19D/saved-models/COV19D_2nd/Modified_Xception.h5")
model.save("/home/idu/Desktop/COV19D/saved-models/COV19D_2nd/Modified_VGG.h5")

# ============ Load Checkpoint ============
 model = keras.models.load_model("/home/idu/Desktop/COV19D/saved-models/Transfer Learning/imageprocessed-Xception.h5")
 # get weights
 modelWeights = model.get_weights()
 # get optimizer state as it was on last epoch
 modelOptimizer = model.optimizer

 # ============ Compile Model ============
 # redefine architecture (newModel=models.Sequential(), etc.)
 newModel= redefine_your_model_architecture()
 #$ newModel= previous model architecture
 # compile
 newModel.compile(optimizer=modelOptimizer,
                  loss = 'binary_crossentropy',
                  metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])
 # set trained weights
 newModel.set_weights(modelWeights)

 # Resume Training if interrupted
 history = newModel.fit(train_generator, 
                        validation_data=val_generator, epochs=30, 
                        callbacks=[callbacks])
 

#Evaluation of the model on the train and validation sets

## Accuracy & Loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

### Precision, Recall, Macro F1 score
val_recall = history.history['val_recall']
print(val_recall)
avg_recall = np.mean(val_recall)
avg_recall

val_precision = history.history['val_precision']
avg_precision = np.mean(val_precision)
avg_precision

Train_accuracy = history.history['accuracy']

epochs = range(1, len(Train_accuracy)+1)
plt.figure(figsize=(12,6))
plt.plot(epochs, val_recall, 'g', label='Validation Recall')
plt.plot(epochs, val_precision, 'b', label='Validation Prcision')
plt.title('Validation recall and Validation Percision')
plt.xlabel('Epochs')
plt.ylabel('Recall and Precision')
plt.legend()
plt.ylim(0,1)

plt.show()

###### The macro F1 score on the validation sex (0.78232)
Macro_F1score = (2*avg_precision*avg_recall)/ (avg_precision + avg_recall)
Macro_F1score


#Making predictions on the test set of unseen images; COV19-CT-DB, ECCV dataset release

## Choosing the directory where the test/validation data is at
folder_path = '/home/idu/Desktop/COV19D/val-processed/covid' # Change as needed

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
results =1
for fldr in os.listdir(folder_path):
   if fldr.startswith("ct"):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        c=load_img(file_path, color_mode='rgb', target_size=(224,224))       
        c=img_to_array(c)
        c= np.expand_dims(c, axis=0)
        c /= 255.0
        result = model.predict(c) #Probability of 1 (non-covid)
        if result > 0.97:  # Class probability threshod is 0.97
           extensions1.append(results)
        else:
           extensions0.append(results)
        if result > 0.90:  # Class probability threshod is 0.90 
           extensions3.append(results)
        else:
           extensions2.append(results) 
        if result > 0.70:   # Class probability threshod is 0.70
           extensions5.append(results)
        else:
           extensions4.append(results)
        if result > 0.40:   # Class probability threshod is 0.40
           extensions7.append(file_path)
        else:
           extensions6.append(results)
        if result > 0.50:   # Class probability threshod is 0.50
           extensions9.append(results)
        else:
           extensions8.append(results)
        if result > 0.15:   # Class probability threshod is 0.15
           extensions11.append(results)
        else:
           extensions10.append(results)  
        if result > 0.05:   # Class probability threshod is 0.05
           extensions13.append(results)
        else:
           extensions12.append(results)
    #print(sub_folder_path, end="\r \n")
    ## The majority voting at Patient's level
    if len(extensions1) >  len(extensions0):
      print(fldr, colored("NON-COVID", 'red'), len(extensions1), "to", len(extensions0))
      noncovidd.append(fldr)  
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions0), "to", len(extensions1))
      covidd.append(fldr)    
    if len(extensions3) >  len(extensions2):
      print (fldr, colored("NON-COVID", 'red'), len(extensions3), "to", len(extensions2))
      noncoviddd.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions2), "to", len(extensions3))
      coviddd.append(fldr)
    if len(extensions5) >  len(extensions4):
      print (fldr, colored("NON-COVID", 'red'), len(extensions5), "to", len(extensions4))
      noncovidddd.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions5), "to", len(extensions4))
      covidddd.append(fldr)
    if len(extensions7) >  len(extensions6):
      print (fldr, colored("NON-COVID", 'red'), len(extensions7), "to", len(extensions6))
      noncoviddddd.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions6), "to", len(extensions7))
      coviddddd.append(fldr)
    if len(extensions9) >  len(extensions8):
      print (fldr, colored("NON-COVID", 'red'), len(extensions9), "to", len(extensions8))
      noncovidd6.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions8), "to", len(extensions9))
      covidd6.append(fldr)
    if len(extensions11) >  len(extensions10):
      print (fldr, colored("NON-COVID", 'red'), len(extensions11), "to", len(extensions10))
      noncovidd7.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions10), "to", len(extensions11))
      covidd7.append(fldr)
    if len(extensions13) > len(extensions12):
      print (fldr, colored("NON-COVID", 'red'), len(extensions13), "to", len(extensions12))
      noncovidd8.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions12), "to", len(extensions13))
      covidd8.append(fldr)
       
    extensions0=[]
    extensions1=[]
    extensions2=[]
    extensions3=[]
    extensions4=[]
    extensions5=[]
    extensions6=[]
    extensions7=[]
    extensions8=[]
    extensions9=[]
    extensions10=[]
    extensions11=[]
    extensions12=[]
    extensions13=[]

#Checking the results
#print(len(covidd))
#print(len(coviddd))
print(len(covidddd))
print(len(coviddddd))
print(len(covidd6))
print(len(covidd7))
#print(len(covidd8))
#print(len(noncovidd))
#print(len(noncoviddd))
print(len(noncovidddd))
print(len(noncoviddddd))
print(len(noncovidd6))
print(len(noncovidd7))
#print(len(noncovidd8))
#print(len(covidd+noncovidd))
#print(len(coviddd+noncoviddd))
print(len(covidddd+noncovidddd))
print(len(coviddddd+noncoviddddd))
print(len(covidd6+noncovidd6))
print(len(covidd7+noncovidd7))
#print(len(covidd8+noncovidd8))


### Saving to csv files format Using Majority Votingat the slice level 0.5 slice level class probability 
import csv

with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncovidd7)

with open('/home/idu/Desktop/covid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(covidd7)

## Using 0.9 Slice level class probability
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncoviddd)

with open('/home/idu/Desktop/ncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(coviddd)

## Using 0.15 Slice level class probability
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncovidd7)

with open('/home/idu/Desktop/covid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(covidd7)
 
 ## Using 0.4 Slice level class probability
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncoviddddd)

with open('/home/idu/Desktop/covid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(coviddddd)

 
## Statistical Analysis of Model Miscalssifications
# Drectly Checking the Images
file_path = '/home/idu/Desktop/COV19D/val-processed/covid/ct_scan168/177.jpg' # Change as neede

c=load_img(file_path, color_mode='rgb', target_size=(224,224))       
c=img_to_array(c)
c= np.expand_dims(c, axis=0)
c /= 255.0
result = model.predict(c) 
if result > 0.4: ## The class probaility threshold
 print('non-covid')
else:
 print('covid')
 
# Studing number of miscalssification in each slice
import os
import csv
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd

folder_path = '/home/idu/Desktop/COV19D/val-processed/covid'  # Change as needed
All_slices = []
slices_extensions = []

# Create a list to store the counts
counts_data = []

# Assuming 'model' is defined before this code
for fldr in os.listdir(folder_path):
    if fldr.startswith("ct"):
        sub_folder_path = os.path.join(folder_path, fldr)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            c = load_img(file_path, color_mode='rgb', target_size=(224, 224))
            c = img_to_array(c)
            c = np.expand_dims(c, axis=0)
            c /= 255.0
            result = model.predict(c)  # Probability of 1 (non-covid)

            # Misclassification Case
            if result > 0.40:
                slices_extensions.append(file_path)
            All_slices.append(file_path)

        misclassified_slices_count = len(slices_extensions)
        all_slices_count = len(All_slices)
        counts_data.append((misclassified_slices_count, all_slices_count))

        print(sub_folder_path, misclassified_slices_count, '/', all_slices_count)

# Save counts_data to a CSV file
csv_file_path = '/home/idu/Desktop/counts_data.csv'  # Change the path as needed
with open(csv_file_path, 'w', newline='') as csv_file:
    fieldnames = ['Number of Misclassified Slices', 'Number of All Slices']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for row in counts_data:
        writer.writerow({'Number of Misclassified Slices': row[0], 'Number of All Slices': row[1]})
        
# Convert the CSV file to Excel
excel_file_path = '//home/idu/Desktop/counts_data.xlsx'  # Change the path as needed
df = pd.read_csv(csv_file_path)
df.to_excel(excel_file_path, index=False)

         
         
### By KENAN MORANI














