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


# In[42]:


pip install --upgrade pywavelets


# In[11]:


import numpy as np
import pywt
import matplotlib.pyplot as plt

# Assuming signals is a list containing gait signals for each patient
num_scales = 8

# Choose scales and wavelet
scales = range(1, 128)
wavelet = 'gaus1'  # Example: Using 'gaus1' wavelet instead of 'bb'

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


# In[12]:


import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming signals is a list containing gait signals for each patient
# Assuming train_data contains the CWT coefficients for each patient

# Choose the output directory for saving spectrogram images
output_directory = r"D:\spectrogram images"

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


# In[13]:


import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming signals is a list containing gait signals for each patient
# Assuming train_data contains the CWT coefficients for each patient

# Choose the output directory for saving spectrogram images
output_directory = r"D:\spect resized"

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


# In[14]:


import os
from PIL import Image

# Set the path to the original images directory
original_directory = r"D:\spect resized"

# Set the path to the resized images directory
resized_directory_224 = r"D:\spect 224"

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


# In[18]:


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
data_dir = r"D:\spect 224"

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


# In[19]:


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


# In[20]:


# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)


# In[21]:


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

# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)



# In[ ]:




