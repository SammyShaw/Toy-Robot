### Toy Robot ### Mobile Net v2 ###

### Cars vs Duplos. MobileNetV2 ###

import os 
os.chdir("C:/Dat_Sci/Data Projects/Toy Robot")

# Import required Packages

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# --------------------------
# 1. Data Preparation
# --------------------------

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128

# Augment Training Data
training_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=10,
    brightness_range=(0.8, 1.3),
    fill_mode='nearest'
)


# No Augmentation for Validation
validation_generator = ImageDataGenerator(rescale=1./255)

# Load Data
training_set = training_generator.flow_from_directory(
    directory=training_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


validation_set = validation_generator.flow_from_directory(
    directory=validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

print(training_set.class_indices)
print(validation_set.class_indices)
# --------------------------
# 2. Model Architecture
# --------------------------

# Load Pretrained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False  # Freeze base layers initially


# Build Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.4),  # Helps prevent overfitting
    Dense(5, activation='softmax')  # Binary classification
])

# Compile Model
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam',
              metrics = ['accuracy'])

# View Model Summary
model.summary()

# --------------------------
# 3. Training Setup
# --------------------------

num_epochs = 20
model_filename = 'models/mobilenetv2_v1.h5'

# Callbacks
save_best_model = ModelCheckpoint(filepath=model_filename,
                                  monitor='val_accuracy',
                                  mode='max',
                                  verbose=1,
                                  save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              min_lr=1e-6,
                              verbose=1)

# Train Model
history = model.fit(training_set,
                    validation_data=validation_set,
                    epochs=num_epochs,
                    callbacks=[save_best_model, reduce_lr])




## Observe Test Set ##

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

model_filename = 'models/mobilenetv2_v1.h5'
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
    class_probs = model.predict(image)[0]  # Get all class probabilities
    predicted_class = np.argmax(class_probs)  # Find class with highest probability
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[predicted_class]  # Probability of predicted class
    
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








# --------------------------------------------
### Unfreeze last layers #####################


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# --------------------------
# 1. Data Preparation
# --------------------------

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128

# Augment Training Data
training_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=10,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

# No Augmentation for Validation
validation_generator = ImageDataGenerator(rescale=1./255)

# Load Data
training_set = training_generator.flow_from_directory(
    directory=training_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

validation_set = validation_generator.flow_from_directory(
    directory=validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# --------------------------
# 2. Model Architecture
# --------------------------

# Load the entire model instead of manually recreating it
model = tf.keras.models.load_model("models/mobilenetv2_two.h5")

# Extract the base MobileNetV2 model
base_model = model.layers[0]  

# Unfreeze last 10 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True  # Unfreeze

# Recompile with a small learning rate for fine-tuning
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-5),  # 🔥 Lower LR for fine-tuning
    metrics=['accuracy']
)

# View Model Summary
model.summary()

# Continue training
fine_tune_epochs = 20  
history_fine = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=fine_tune_epochs,
    callbacks=[save_best_model, reduce_lr]
)






## Observe Test Set ##

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

model_filename = 'models/mobilenetv2_two.h5'
img_width = 128
img_height = 128
labels_list = ['cars', 'duplos']

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
    class_probs = model.predict(image)[0][0]
    predicted_class = 1 if class_probs > 0.5 else 0
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs
    
    return predicted_label, predicted_prob

# loop through test data

source_dir = 'data/test/'
folder_names = ['cars', 'duplos']
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








import os

# Specify the directory you want to check
directory = 'data/training'  # Change this to your desired directory

# Create a dictionary to count file types
file_types = {}

# Walk through the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Get the file extension
        ext = os.path.splitext(file)[1].lower()
        if ext in file_types:
            file_types[ext] += 1
        else:
            file_types[ext] = 1

# Print out the results
print("Found the following file types:")
for ext, count in file_types.items():
    print(f"{ext}: {count} files")
    
from PIL import Image
def convert_to_jpg(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in ['.png', '.jpeg']:
                img = Image.open(file_path).convert('RGB')  # Convert to RGB to avoid mode errors
                new_file_path = os.path.splitext(file_path)[0] + '.jpg'
                img.save(new_file_path, 'JPEG')
                os.remove(file_path)  # Remove the original file
                print(f"Converted {file} to {new_file_path}")

# Run conversion on your training data directory
convert_to_jpg('data/training')



# Directories to check
directories = ['data/training', 'data/validation']
corrupted_files = []

def check_corrupted_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                try:
                    # Try opening and loading the image
                    img = Image.open(file_path)
                    img.verify()  # Verify that it is a proper image
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted file detected: {file_path}")
                    corrupted_files.append(file_path)

# Run the check for each directory
for dir in directories:
    check_corrupted_images(dir)

# Save results to a file
if corrupted_files:
    with open('corrupted_images.txt', 'w') as f:
        for file in corrupted_files:
            f.write(file + '\n')
    print(f"\nFound {len(corrupted_files)} corrupted images. See 'corrupted_images.txt' for details.")
else:
    print("\nNo corrupted images found.")
    