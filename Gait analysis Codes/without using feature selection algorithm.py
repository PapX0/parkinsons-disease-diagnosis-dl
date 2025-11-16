#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pywt
import os


# In[2]:


# Read demographic file from dataset and create patient dictionary
patient_dict = {}

with open(r"D:\Newfolder\s7\project\gait dataset\gait-in-parkinsons-disease-1.0.0\demographics.txt", 'r') as file:
   data = file.read().splitlines()
   # Skip the header
   data.pop(0)

   for line in data:
       linedata = line.split()
       # Check if linedata has at least 9 elements before accessing index 8
       if len(linedata) >= 9 and linedata[8] == "NaN":
           patient_dict[linedata[0]] = 0.0
       elif len(linedata) >= 9:
           patient_dict[linedata[0]] = float(linedata[8])

# Print the resulting dictionary
print(patient_dict)



# In[ ]:





# In[3]:


# Read in data from datafiles and store in lists
signals = []
labels = []

for filename in os.listdir(r"D:\Newfolder\s7\project\gait dataset\gait-in-parkinsons-disease-1.0.0\data"):
  with open(os.path.join(r"D:\Newfolder\s7\project\gait dataset\gait-in-parkinsons-disease-1.0.0\data", filename), 'r', encoding='latin-1') as file:
      data = file.read().splitlines()
      sig1 = []
      sig2 = []
      sig3 = []
      sig4 = []
      sig5 = []
      sig6 = []
      sig7 = []
      sig8 = []

      for line in data:
          linedata = line.split()

          # Add a check to ensure that the line contains numeric data
          if len(linedata) >= 17:
              try:
                  sig1.append(float(linedata[1]))
                  sig2.append(float(linedata[4]))
                  sig3.append(float(linedata[5]))
                  sig4.append(float(linedata[8]))
                  sig5.append(float(linedata[9]))
                  sig6.append(float(linedata[12]))
                  sig7.append(float(linedata[13]))
                  sig8.append(float(linedata[16]))
              except ValueError as e:
                  print(f"Error processing line in file {filename}: {e}")

      tempsig = [sig1, sig2, sig3, sig4, sig5, sig6, sig7, sig8]
      signals.append(tempsig)
      labels.append(patient_dict.get(filename[:6], 0.0))  # Assuming 0.0 if no label found

# Print labels for verification
print(labels)


# In[4]:


import numpy as np

# Iterate over each patient's set of signals
for patient_signals in signals:
   # Iterate over each of the 306 elements
   for i in range(len(patient_signals)):
       # Calculate mean and standard deviation for normalization
       mean_value = np.mean(patient_signals[i])
       std_value = np.std(patient_signals[i])

       # Normalize and zero-center each signal element
       patient_signals[i] = (patient_signals[i] - mean_value) / std_value

# Print the first few elements of the normalized signals for verification
for i in range(min(5, len(signals))):
   print(signals[i])


# In[5]:


max_val = 1000000
max_index = 0

for j, i in enumerate(signals):
   if len(i[0]) > 0 and len(i[0]) < max_val:
       max_val = len(i[0])
       max_index = j

print(max_val, max_index - 1)


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


step = 0.01
arr = [0]
for i in range(999):
    arr.append(arr[i]+step)
plt.figure(figsize=(40,40))
plt.plot(arr, signals[1][2][0:1000])

plt.show() 


# In[8]:


for i in range(len(signals)):
   for j in range(len(signals[i])):
       signals[i][j] = signals[i][j][0:1000]


# In[9]:


for i in signals:
   for j in i:
       print(len(j))


# In[ ]:





# In[10]:


import numpy as np
import pywt
import matplotlib.pyplot as plt

# Assuming signals is a list containing gait signals for each patient
num_scales = 8

# Choose scales and wavelet
scales = range(1, 128)
wavelet = 'morl'

# Create an empty ndarray for the CWT coefficients
train_data = np.ndarray(shape=(len(signals), 127, 1000, num_scales), dtype='float32')

# Apply CWT to each signal
for j, signal in enumerate(signals):
  temp_coeffs = []
  for i in range(num_scales):
      try:
          coeffs, freq = pywt.cwt(signal[i], scales, wavelet, 1)
          temp_coeffs.append(coeffs)
      except ValueError as e:
          print(f"Error processing signal {j}, scale {i}: {e}")
          # If there's an error, fill with zeros
          temp_coeffs.append(np.zeros_like(signal[i]))

  # Fill train_data with the coefficients
  for k in range(len(temp_coeffs)):
      for a in range(len(temp_coeffs[0])):
          for b in range(len(temp_coeffs[0][0])):
              train_data[j, a, b, k] = temp_coeffs[k][a][b]

# Plot the CWT result for a specific patient (e.g., the first patient)
coeffs, freq = pywt.cwt(signals[0][0], scales, wavelet, 1)
plt.subplots(figsize=(30, 40))
plt.imshow(coeffs, cmap='magma', interpolation='nearest', aspect='auto')
plt.title('Continuous Wavelet Transform')
plt.xlabel('Time')
plt.ylabel('Scale')
plt.colorbar(label='Magnitude')
plt.show()


# In[18]:


import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming signals is a list containing gait signals for each patient
# Assuming train_data contains the CWT coefficients for each patient

# Choose the output directory for saving spectrogram images
output_directory = r"D:\Newfolder\s7\spect"

# Create the output directories for Co and Pt if they don't exist
os.makedirs(os.path.join(output_directory, 'Co'), exist_ok=True)
os.makedirs(os.path.join(output_directory, 'Pt'), exist_ok=True)

# Iterate over patients
for patient_index, patient_name in enumerate(os.listdir(r"D:\Newfolder\s7\project\gait dataset\gait-in-parkinsons-disease-1.0.0\data")):
   plt.figure(figsize=(15, 10))
   
   # Plot the spectrogram-like representation
   for signal_index in range(num_scales):
       plt.imshow(np.abs(train_data[patient_index, :, signal_index, :]), cmap='hot', aspect='auto', extent=[0, 1000, 1, 128])
   
   # Set title, labels, and colorbar
   plt.title(f'CWT Coefficients - {patient_name}')
   plt.xlabel('Time')
   plt.ylabel('Scale')
   plt.colorbar(label='Magnitude')

   plt.tight_layout()
   
   # Save the spectrogram image with the file name including patient details
   label = labels[patient_index]
   folder_name = 'Co' if label == 0.0 else 'Pt'
   image_filename = f"spectrogram_{patient_name}_cwt.png"
   image_path = os.path.join(output_directory, folder_name, image_filename)
   plt.savefig(image_path)
   
   # Close the plot to avoid displaying multiple plots
   plt.close()

print("Spectrogram images saved to:", output_directory)


# In[19]:


import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming signals is a list containing gait signals for each patient
# Assuming train_data contains the CWT coefficients for each patient

# Choose the output directory for saving spectrogram images
output_directory = r"D:\Newfolder\s7\spect resized"

# Create the output directories for Co and Pt if they don't exist
os.makedirs(os.path.join(output_directory, 'Co'), exist_ok=True)
os.makedirs(os.path.join(output_directory, 'Pt'), exist_ok=True)

# Iterate over patients
for patient_index, patient_name in enumerate(os.listdir(r"D:\Newfolder\s7\project\gait dataset\gait-in-parkinsons-disease-1.0.0\data")):
   plt.figure(figsize=(15, 10))
   
   # Plot the spectrogram-like representation
   for signal_index in range(num_scales):
       plt.imshow(np.abs(train_data[patient_index, :, signal_index, :]), cmap='hot', aspect='auto', extent=[0, 1000, 1, 128])
   
   # Remove axis labels and ticks
   plt.axis('off')
   
   # Save the spectrogram image with the file name including patient details
   label = labels[patient_index]
   folder_name = 'Co' if label == 0.0 else 'Pt'
   image_filename = f"spectrogram_{patient_name}_cwt.png"
   image_path = os.path.join(output_directory, folder_name, image_filename)
   plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
   
   # Close the plot to avoid displaying multiple plots
   plt.close()

print("Spectrogram images saved to:", output_directory)


# In[20]:


import os
from PIL import Image

# Set the path to the original images directory
original_directory = r"D:\Newfolder\s7\spect resized"

# Set the path to the resized images directory
resized_directory_224 = r"D:\Newfolder\s7\spect 224"

# Create the output directories for Co and Pt in the resized directory if they don't exist
os.makedirs(os.path.join(resized_directory_224, 'Co'), exist_ok=True)
os.makedirs(os.path.join(resized_directory_224, 'Pt'), exist_ok=True)

# Iterate through Co and Pt folders in the original directory
for folder_name in ['Co', 'Pt']:
   original_folder_path = os.path.join(original_directory, folder_name)
   resized_folder_path_224 = os.path.join(resized_directory_224, folder_name)
   
   # Iterate through image files in the folder
   for filename in os.listdir(original_folder_path):
       if filename.endswith(".png"):
           # Construct the full path to the original image
           original_image_path = os.path.join(original_folder_path, filename)
           
           # Open the original image
           img = Image.open(original_image_path)
           
           # Resize the image to 224x224
           img_resized_224 = img.resize((224, 224), Image.LANCZOS)
           
           # Save the resized image
           resized_image_path_224 = os.path.join(resized_folder_path_224, f"resized_{filename}")
           img_resized_224.save(resized_image_path_224)

print("Images resized to 224x224 and saved to:", resized_directory_224)


# In[12]:


import os
from sklearn.model_selection import train_test_split

# Set the path to the parent directory containing 'Co' and 'Pt' folders
parent_directory = r"D:\Newfolder\s7\project\spectrogram images\images with noramalisation\spect 224"

# Create lists to store image paths and corresponding labels
image_paths = []
labels = []

# Iterate through Co and Pt folders
for folder_name in ['Co', 'Pt']:
    folder_path = os.path.join(parent_directory, folder_name)
    
    # Iterate through image files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # Construct the full path to the image
            image_path = os.path.join(folder_path, filename)
            
            # Append the image path to the list
            image_paths.append(image_path)
            
            # Assign label based on folder name
            label = 0 if folder_name == 'Co' else 1
            labels.append(label)

# Split the data into training and testing sets
train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Print the number of samples in each set
print("Number of training samples:", len(train_image_paths))
print("Number of testing samples:", len(test_image_paths))


# In[14]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, labels):
   images = []
   for path in image_paths:
       img = load_img(path, target_size=(224, 224))
       img_array = img_to_array(img)
       images.append(img_array)

   # Convert the list to a NumPy array
   images = np.array(images)

   # Normalize pixel values to be between 0 and 1
   images = images / 255.0

   # Convert labels to categorical (one-hot encoding)
   labels = to_categorical(labels)

   return images, labels

# Load and preprocess training and testing images
X_train, y_train = load_and_preprocess_images(train_image_paths, train_labels)
X_test, y_test = load_and_preprocess_images(test_image_paths, test_labels)

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_binary)
print(f"Test set accuracy: {accuracy * 100:.2f}%")


# In[1]:


import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define the image size and batch size
img_size = (224, 224)
batch_size = 32
data_dir = r"D:\Newfolder\s7\project\spectrogram images\images with noramalisation\spect 224"

# Create a data generator for training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=img_size,
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=img_size,
  batch_size=batch_size)


# In[29]:


class_names = train_ds.class_names
print(class_names)


# In[30]:


print(len(train_ds))


# In[43]:


# Import necessary libraries
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Create the ResNet-50 base model (without the top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create the model by adding custom top layers
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[44]:


# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)


# In[46]:


from sklearn.metrics import classification_report
import numpy as np

# Evaluate the model on the validation set
val_pred = model.predict(val_ds)
val_pred_binary = np.round(val_pred)

# Get true labels from the validation set
val_true = []
for _, labels in val_ds:
   val_true.extend(labels.numpy())

# Convert the true labels to a numpy array
val_true = np.array(val_true)

# Generate classification report
class_report = classification_report(val_true, val_pred_binary, target_names=['Negative', 'Positive'])
print("Classification Report:\n", class_report)

# Extract precision, recall, and F1 score from the classification report
precision_recall_f1 = [float(value) for value in class_report.split()[-3:]]

precision, recall, f1_score = precision_recall_f1
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# In[16]:


# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)


# In[41]:


from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Create the InceptionV3 base model (without the top layer)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create the model by adding custom top layers
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[42]:


# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)


# In[31]:


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)

# Evaluate the model on the validation set
val_pred = model.predict(val_ds)
val_pred_binary = np.round(val_pred)

# Get true labels from validation set
val_true = val_ds.classes

# Calculate confusion matrix
conf_matrix = confusion_matrix(val_true, val_pred_binary)

# Calculate precision, recall, and F1 score
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Confusion Matrix:")
print(conf_matrix)


# In[20]:


# Import necessary libraries
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Create the VGG16 base model (without the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create the model by adding custom top layers
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[21]:


# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)


# In[2]:


# Import necessary libraries
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Create the VGG16 base model (without the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create the model by adding custom top layers
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
#history = model.fit(train_ds, epochs=30, validation_data=val_ds)



# In[3]:


# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)



# In[4]:


#transfer
# Import necessary libraries
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Create the VGG16 base model (without the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create the model by adding custom top layers
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Unfreeze some of the layers in the base model for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with fine-tuning
history = model.fit(train_ds, epochs=15, validation_data=val_ds)


# In[37]:


import numpy as np

# Evaluate the model on the validation set
val_pred = model.predict(val_ds)
val_pred_binary = np.round(val_pred)

# Get true labels from the validation set
val_true = []
for images, labels in val_ds:
   val_true.extend(labels.numpy())

# Convert the true labels to a numpy array
val_true = np.array(val_true)

# Generate classification report
class_report = classification_report(val_true, val_pred_binary, target_names=['Negative', 'Positive'])
print("Classification Report:\n", class_report)

# Parse the classification report string into a dictionary
report_dict = classification_report(val_true, val_pred_binary, target_names=['Negative', 'Positive'], output_dict=True)

# Extract precision, recall, and F1 score from the dictionary
precision = report_dict['Positive']['precision']
recall = report_dict['Positive']['recall']
f1_score = report_dict['Positive']['f1-score']

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# In[38]:


from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Build an improved AlexNet-like model
model = Sequential()

# Layer 1
model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# Layer 2
model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# Layer 3
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))

# Layer 4
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))

# Layer 5
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# Global Average Pooling
model.add(GlobalAveragePooling2D())

# Dense Layers with Dropout
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()


# In[39]:


# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)


# In[40]:


from sklearn.metrics import classification_report
import numpy as np

# Evaluate the model on the validation set
val_pred = model.predict(val_ds)
val_pred_binary = np.round(val_pred)

# Get true labels from the validation set
val_true = []
for images, labels in val_ds:
    val_true.extend(labels.numpy())

# Convert the true labels to a numpy array
val_true = np.array(val_true)

# Generate classification report
class_report = classification_report(val_true, val_pred_binary, target_names=['Negative', 'Positive'])
print("Classification Report:\n", class_report)

# Parse the classification report string into a dictionary
report_dict = classification_report(val_true, val_pred_binary, target_names=['Negative', 'Positive'], output_dict=True)

# Extract precision, recall, and F1 score from the dictionary
precision = report_dict['Positive']['precision']
recall = report_dict['Positive']['recall']
f1_score = report_dict['Positive']['f1-score']

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# In[14]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, labels):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)

    # Convert the list to a NumPy array
    images = np.array(images)

    # Normalize pixel values to be between 0 and 1
    images = images / 255.0

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels)

    return images, labels

# Load and preprocess training and testing images
X_train, y_train = load_and_preprocess_images(train_image_paths, train_labels)
X_test, y_test = load_and_preprocess_images(test_image_paths, test_labels)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Build the new model on top of the pre-trained VGG16
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))



# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_binary)
print(f"Test set accuracy: {accuracy * 100:.2f}%")


# In[ ]:





# In[ ]:





# In[15]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score
import numpy as np

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, labels):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)

    # Convert the list to a NumPy array
    images = np.array(images)

    # Normalize pixel values to be between 0 and 1
    images = images / 255.0

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels)

    return images, labels

# Load and preprocess training and testing images
X_train, y_train = load_and_preprocess_images(train_image_paths, train_labels)
X_test, y_test = load_and_preprocess_images(test_image_paths, test_labels)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Build the new model on top of the pre-trained VGG16
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))


# Define the learning rate scheduler
initial_learning_rate = 0.001

def lr_scheduler(epoch, lr):
    return lr * 0.9  # Adjust the multiplier as needed

lr_callback = LearningRateScheduler(lr_scheduler)

# Define the optimizer with the learning rate
optimizer = Adam(learning_rate=initial_learning_rate)

# Compile the model with the optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with the learning rate scheduler callback
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[lr_callback])

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_binary)
print(f"Test set accuracy: {accuracy * 100:.2f}%")


# In[30]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, labels):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)

    # Convert the list to a NumPy array
    images = np.array(images)

    # Normalize pixel values to be between 0 and 1
    images = images / 255.0

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels)

    return images, labels

# Load and preprocess training and testing images
X_train, y_train = load_and_preprocess_images(train_image_paths, train_labels)
X_test, y_test = load_and_preprocess_images(test_image_paths, test_labels)

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Build the new model on top of the pre-trained ResNet50
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_binary)
print(f"Test set accuracy: {accuracy * 100:.2f}%")


# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

# Define AlexNet architecture
def AlexNet(input_shape, num_classes):
    model = models.Sequential()
    
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, labels):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=(227, 227))  # AlexNet input size is 227x227
        img_array = img_to_array(img)
        images.append(img_array)

    # Convert the list to a NumPy array
    images = np.array(images)

    # Normalize pixel values to be between 0 and 1
    images = images / 255.0

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels)

    return images, labels

# Load and preprocess training and testing images
X_train, y_train = load_and_preprocess_images(train_image_paths, train_labels)
X_test, y_test = load_and_preprocess_images(test_image_paths, test_labels)

# Load the custom AlexNet model
input_shape = (227, 227, 3)  # AlexNet input size is 227x227
num_classes = 2  # Change this based on your classification task
base_model = AlexNet(input_shape, num_classes)

# Compile the model
base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
base_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred = base_model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_binary)
print(f"Test set accuracy: {accuracy * 100:.2f}%")


# In[ ]:




