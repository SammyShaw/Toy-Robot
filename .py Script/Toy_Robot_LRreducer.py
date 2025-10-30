### Toy Robot - LR Reducer

# Dropout
# Augmented (min)
# LR Reducer 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from keras_tuner.engine.hyperparameters import HyperParameters

import os 
os.chdir("C:/Dat_Sci/Data Projects/Toy Robot")

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

# all training data image generator

training_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15, 
    width_shift_range=0.15,  # Reduce shift (was 0.2)
    height_shift_range=0.15,
    zoom_range=0.15,  # Keep small zoom (was 0.1)
    horizontal_flip=True,  # Keep flipping
    shear_range=5,
    brightness_range=(0.9, 1.2),  # Less extreme brightness (was 0.5–1.5)
    fill_mode='nearest'
)


training_set = training_generator.flow_from_directory(
    directory = training_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


validation_generator = ImageDataGenerator(rescale = 1./255)

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
model.add(Dropout(0.3))
model.add(Activation('softmax')) 

# Compile
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Summary
model.summary()

#########################################################################
# Train Our Network!
#########################################################################

# training parameters

num_epochs = 50
model_filename = 'models/Toy_Robot_LRreducer_v03.h5'

# callbacks
save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1, 
                                  save_best_only = True)


# reduce learning when validation loss stops improving
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',  # Monitor validation loss
                              factor=0.5,         # Reduce LR by half
                              patience=4,         # Wait 3 epochs before reducing
                              min_lr=1e-6,        # Minimum learning rate
                              verbose=1)          # Print updates
# train the network

history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model, reduce_lr])


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

model_filename = 'models/Toy_Robot_LRreducer_v03.h5'
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



# errors analysis

print(predictions_df[["predicted_probability", "correct"]])

misclassifications = predictions_df[predictions_df['correct'] == 0]  # Get all wrong predictions
print(misclassifications['actual_label'].value_counts())  # Count misclassifications per class

print(misclassifications['predicted_label'].value_counts())  # Count most common wrong guesses

high_conf_misclass = predictions_df[(predictions_df['correct'] == 0) & (predictions_df['predicted_probability'] > 0.6)]
print(high_conf_misclass)

print(misclassifications[["predicted_probability", "actual_label", "predicted_label"]])







# Confirm structure of data 

print("Detected classes:", training_set.class_indices)
print("Number of classes detected:", training_set.num_classes)


class_counts = np.bincount(training_set.classes)
print("Training set class distribution:", class_counts)


batch = next(iter(training_set))
print("Input shape:", batch[0].shape)
print("Label shape:", batch[1].shape)

