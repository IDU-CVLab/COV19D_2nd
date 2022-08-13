#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 15:50:19 2021

@author: idu
"""

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

##########################Using image datagenerator for Generating data with rescaling and binary labels from the images (rgb images)
batch_size = 128
SIZE = 224  ## Resizing images to 224x224

train_datagen = ImageDataGenerator(rescale=1./255)
                                   
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train/', 
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        color_mode='rgb', 
        class_mode='binary')  

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/validation/',  
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        color_mode='rgb',
        class_mode='binary')

##########################################################################################################3Cropping
###################################### Transfer Learning Models####################################333

### Using pretrained Xception model
Model_Xcep = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
Model_VGG = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))

for layer in Model_Xcep.layers:
	layer.trainable = False
    
for layer in Model_VGG.layers:
	layer.trainable = False
    
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

model = tf.keras.Sequential([
    Model_VGG, 
    tf.keras.layers.GlobalAveragePooling2D(), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    
    
#model = tf.keras.models.load_model ("Modified_Xception1.h5")    
model.summary()  

# Adding callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("Modified_VGG1.h5", save_best_only=True, verbose = 0),
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
 model = keras.models.load_model("/home/idu/Modified_VGG1.h5")
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

 # ============ Resume Training ============
 history = newModel.fit(train_generator, 
                        validation_data=val_generator, epochs=30, 
                        callbacks=[callbacks])
 

######### Evaluation of the model on the train and validation sets

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

###############
#^^^^^^^^^^^^^^^^^^ Testing the model on the Test Set for COV19-CT-DB 2nd Chellenge
#^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
####^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

############################## Making Predictions at patient level on the test set

## Choosing the directory where the test/validation data is at
folder_path = '/home/idu/Desktop/COV19D/validation/non-covid'
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
        result = model.predict_proba(c) #Probability of 1 (non-covid)
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
           extensions7.append(results)
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
print(len(covidd))
print(len(coviddd))
print(len(covidddd))
print(len(coviddddd))
print(len(covidd6))
print(len(covidd7))
print(len(covidd8))
print(len(noncovidd))
print(len(noncoviddd))
print(len(noncovidddd))
print(len(noncoviddddd))
print(len(noncovidd6))
print(len(noncovidd7))
print(len(noncovidd8))
print(len(covidd+noncovidd))
print(len(coviddd+noncoviddd))
print(len(covidddd+noncovidddd))
print(len(coviddddd+noncoviddddd))
print(len(covidd6+noncovidd6))
print(len(covidd7+noncovidd7))
print(len(covidd8+noncovidd8))


### Saving to csv files format
############## Using Majority Votingat the slice level
########### 0.5 slice level class probability 
import csv

with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncovidd6)

with open('/home/idu/Desktop/covid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(covidd6)

############## 0.9 Slice level class probability
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncoviddd)

with open('/home/idu/Desktop/ncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(coviddd)

############ 0.15 Slice level class probability
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncovidd7)

with open('/home/idu/Desktop/covid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(covidd7)
 
 ### KENAN MORANI - THE END
 
 
 ############## 0.4 Slice level class probability
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncoviddddd)

with open('/home/idu/Desktop/covid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(coviddddd)














