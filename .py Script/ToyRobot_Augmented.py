### Toy Robot Augmented ###

### Toy Robot - Cars vs. Duplos - Basic ###

import os 
os.chdir("C:/Dat_Sci/Data Projects/Toy Robot")

# Import required Packages

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


#########################################################################
# Set Up flow For Training & Validation data
#########################################################################

# data flow parameters

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 5


# image generators

training_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20, 
    width_shift_range=0.2,  # Reduce shift (was 0.2)
    height_shift_range=0.2,
    zoom_range=0.2,  # Keep small zoom (was 0.1)
    horizontal_flip=True,  # Keep flipping
    brightness_range=(0.7, 1.3),  # Less extreme brightness (was 0.5–1.5)
    fill_mode='nearest'
)


validation_generator = ImageDataGenerator(rescale = 1./255)

# image flows

training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

print(training_set.class_indices)
print(validation_set.class_indices)
#########################################################################
# Network Architecture
#########################################################################

# network architecture

model = Sequential()

# 1st convolutional layer (default stride = 1)
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

# 2nd layer
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

# flatten
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))

# output layer
model.add(Dense(num_classes))
model.add(Activation('softmax')) # multi-category output (6 fruits)

# compile network

model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam',
              metrics = ['accuracy'])

# view network architecture

model.summary()

#########################################################################
# Train Our Network!
#########################################################################

# training parameters

num_epochs = 50
model_filename = 'models/toy_robot_augmented_v02.h5'

# callbacks

save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1, 
                                  save_best_only = True)

# train the network

history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])


#########################################################################
# Visualise Training & Validation Performance
#########################################################################

import matplotlib.pyplot as plt

# plot validation results
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# get best epoch performance for validation accuracy
max(history.history['val_accuracy'])


#########################################################################
# Make Predictions On New Data (Test Set)
#########################################################################

# import required packages

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from os import listdir

# parameters for prediction

model_filename = 'models/toy_robot_augmented_v02.h5'
img_width = 128
img_height = 128
labels_list = ['bananagrams', 'brios', 'cars', 'duplos', 'magnatiles']

# load model

model = load_model(model_filename)

# image pre-processing function

def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image * (1./255)
    
    return image
    

# image prediction function

def make_prediction(image):
    class_probs = model.predict(image)[0]
    predicted_class = np.argmax(class_probs)  # ✅ Get the index of the highest probability
    predicted_label = labels_list[predicted_class]  # ✅ Map index to label
    predicted_prob = class_probs[predicted_class]  # ✅ Get the probability of the predicted class
    return predicted_label, predicted_prob

# loop through test data

source_dir = 'data/test/'
folder_names = ['bananagrams', 'brios', 'cars', 'duplos', 'magnatiles']
actual_labels = []
predicted_labels = []
predicted_probabilities = []
filenames = []

for folder in folder_names: 
    
    images = listdir(source_dir + '/' + folder)
    
    for image in images: 
        
        processed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_probability = make_prediction(processed_image)
        
        actual_labels.append(folder)
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(predicted_probability)
        filenames.append(image)
        
        
# create dataframe to analyse

predictions_df = pd.DataFrame({"actual_label" : actual_labels,
                               "predicted_label" : predicted_labels, 
                               "predicted_probability" : predicted_probabilities,
                               "filename" : filenames})

predictions_df['correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_label'], 1, 0)


# overall test set accuracy

test_set_accuracy = predictions_df['correct'].sum() / len(predictions_df)
# or print(predictions_df['correct'].mean())
print(test_set_accuracy)
        

# confusion matrix (raw numbers)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'])
print(confusion_matrix)

# confusion matrix (percentages)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)






