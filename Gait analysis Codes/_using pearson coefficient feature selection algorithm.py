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


import os
import numpy as np
from scipy.stats import pearsonr

# Define the dataset path
dataset_path = r"D:\Newfolder\s7\project\gait dataset\gait-in-parkinsons-disease-1.0.0\data"

# Define the modified calculate_correlation function
def calculate_correlation(signal_data, labels):
   if isinstance(labels, float):
       labels = np.full(signal_data.shape[0], labels)
   correlations = [pearsonr(signal_data[:, i], labels)[0] for i in range(signal_data.shape[1])]
   return correlations

# Initialize a list to store selected signals for each file
selected_signals = []

# Iterate over each file in the dataset path
for filename in os.listdir(dataset_path):
   # Read the file
   with open(os.path.join(dataset_path, filename), 'r', encoding='latin-1') as file:
       data = file.read().splitlines()
       
       # Initialize signals for the current file
       signals = []
       
       # Iterate over each line in the file
       for line in data:
           linedata = line.split()

           # Add a check to ensure that the line contains numeric data
           if len(linedata) >= 19:
               try:
                   # Extracting the relevant columns (assuming columns 2 to 17 are features)
                   features = [float(linedata[i]) for i in range(1, 17)]
                   # Append features to the signals list
                   signals.append(features)
                   
               except ValueError as e:
                   print(f"Error processing line in file {filename}: {e}")

       # Convert signals to a numpy array
       signals = np.array(signals)
       
       # Get labels for the current patient from patient_dict
       patient_id = filename[:6]
       labels = patient_dict.get(patient_id, 0.0)  # Assuming 0.0 if no label found
       
       # Calculate correlation coefficients
       correlations = calculate_correlation(signals, labels)
       
       # Select the top 8 columns with the highest absolute correlation values
       selected_columns_indices = np.argsort(np.abs(correlations))[-8:]
       selected_signals.append(signals[:, selected_columns_indices])
       
       # Print selected column indices for verification
       print(f"Selected columns for {filename}: {selected_columns_indices}")

# Print selected_signals for verification
print(selected_signals)


# In[4]:


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
                 sig1.append(float(linedata[8]))
                 sig2.append(float(linedata[9]))
                 sig3.append(float(linedata[10]))
                 sig4.append(float(linedata[11]))
                 sig5.append(float(linedata[12]))
                 sig6.append(float(linedata[13]))
                 sig7.append(float(linedata[14]))
                 sig8.append(float(linedata[15]))
             except ValueError as e:
                 print(f"Error processing line in file {filename}: {e}")

     tempsig = [sig1, sig2, sig3, sig4, sig5, sig6, sig7, sig8]
     signals.append(tempsig)
     labels.append(patient_dict.get(filename[:6], 0.0))  # Assuming 0.0 if no label found

# Print labels for verification
print(labels)


# In[7]:


max_val = 1000000
max_index = 0

for j, i in enumerate(signals):
   if len(i[0]) > 0 and len(i[0]) < max_val:
       max_val = len(i[0])
       max_index = j

print(max_val, max_index - 1)


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


step = 0.01
arr = [0]
for i in range(999):
    arr.append(arr[i]+step)
plt.figure(figsize=(40,40))
plt.plot(arr, signals[1][2][0:1000])

plt.show() 


# In[10]:


for i in range(len(signals)):
   for j in range(len(signals[i])):
       signals[i][j] = signals[i][j][0:1000]


# In[11]:


for i in signals:
   for j in i:
       print(len(j))


# In[12]:


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


# In[13]:


import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming signals is a list containing gait signals for each patient
# Assuming train_data contains the CWT coefficients for each patient

# Choose the output directory for saving spectrogram images
output_directory = r"D:\Newfolder\s7\project\spectrogram images\using feature slection algorithm\spect"

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


# In[14]:


import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming signals is a list containing gait signals for each patient
# Assuming train_data contains the CWT coefficients for each patient

# Choose the output directory for saving spectrogram images
output_directory = r"D:\Newfolder\s7\project\spectrogram images\using feature slection algorithm\spect resized"

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


# In[15]:


import os
from PIL import Image

# Set the path to the original images directory
original_directory = r"D:\Newfolder\s7\project\spectrogram images\using feature slection algorithm\spect resized"

# Set the path to the resized images directory
resized_directory_224 = r"D:\Newfolder\s7\project\spectrogram images\using feature slection algorithm\spect 224"

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


# In[17]:


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
data_dir = r"D:\Newfolder\s7\project\spectrogram images\using feature slection algorithm\spect 224"

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


# In[18]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
   plt.xlabel('Predicted labels')
   plt.ylabel('True labels')
   plt.title('Confusion Matrix')
   plt.show()

def plot_roc_curve(y_true, y_pred_probs):
   fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
   roc_auc = auc(fpr, tpr)

   plt.figure()
   lw = 2
   plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
   plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic (ROC) Curve')
   plt.legend(loc="lower right")
   plt.show()

def create_resnet50_model(num_classes):
   model = resnet50(pretrained=True)
   
   # Modify the last fully connected layer for binary classification
   in_features = model.fc.in_features
   model.fc = nn.Linear(in_features, num_classes)

   return model

def train_and_evaluate(data_path, split_ratio=0.8, num_epochs=15):
   # Image dimensions
   height, width = 224, 224  # ResNet50 expects input size (224, 224)

   # Data transformation
   transform = transforms.Compose([
       transforms.Resize((height, width)),
       transforms.ToTensor(),
   ])

   # Create dataset
   dataset = ImageFolder(root=data_path, transform=transform)

   # Split dataset into training and validation sets
   total_size = len(dataset)
   train_size = int(split_ratio * total_size)
   val_size = total_size - train_size

   train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

   # Create loaders
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

   # Create ResNet50 model
   num_classes = 2  # Adjust according to your number of classes
   model = create_resnet50_model(num_classes)

   model.train()

   # Loss function and optimizer
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # Lists to store the learning curve data
   train_losses = []
   train_accuracies = []
   val_accuracies = []

   # Training loop
   for epoch in range(num_epochs):
       # Training
       model.train()
       running_loss = 0.0
       correct_train = 0
       total_train = 0
       for inputs, labels in train_loader:
           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           running_loss += loss.item()

           _, predicted = torch.max(outputs.data, 1)
           total_train += labels.size(0)
           correct_train += (predicted == labels).sum().item()

       average_train_loss = running_loss / len(train_loader)
       train_losses.append(average_train_loss)
       train_accuracy = correct_train / total_train
       train_accuracies.append(train_accuracy)

       # Validation
       model.eval()
       correct_val = 0
       total_val = 0
       y_pred_probs = []
       y_true = []
       y_pred = []
       with torch.no_grad():
           for inputs, labels in val_loader:
               outputs = model(inputs)
               _, predicted = torch.max(outputs.data, 1)
               total_val += labels.size(0)
               correct_val += (predicted == labels).sum().item()
               y_true.extend(labels.numpy())
               y_pred.extend(predicted.numpy())
               y_pred_probs.extend(torch.softmax(outputs, 1)[:, 1].tolist())  # Probability for class 1 (positive)

       val_accuracy = correct_val / total_val
       val_accuracies.append(val_accuracy)

       # Print training loss and validation accuracy
       print(f'Epoch {epoch + 1}/{num_epochs}, '
             f'Training Loss: {average_train_loss:.4f}, '
             f'Training Accuracy: {train_accuracy:.4f}, '
             f'Validation Accuracy: {val_accuracy:.4f}')

   # Plot the learning curve
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(train_losses, label='Training Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()

   plt.subplot(1, 2, 2)
   plt.plot(train_accuracies, label='Training Accuracy')
   plt.plot(val_accuracies, label='Validation Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()

   plt.show()

   # Print final training and validation accuracy
   print(f'Final Training Accuracy: {train_accuracies[-1]:.4f}')
   print(f'Final Validation Accuracy: {val_accuracies[-1]:.4f}')

   # Evaluation
   y_true = np.array(y_true)
   y_pred = np.array(y_pred)
   y_pred_probs = np.array(y_pred_probs)

   # Print predicted class values and true class values
   print("Predicted Class Values:", y_pred)
   print("True Class Values:", y_true)

   # Print confusion matrix and classification report
   print(classification_report(y_true, y_pred))
   plot_confusion_matrix(y_true, y_pred)

   # Calculate and print specificity and sensitivity
   tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
   sensitivity = tp / (tp + fn)
   specificity = tn / (tn + fp)
   print(f'Sensitivity: {sensitivity:.4f}')
   print(f'Specificity: {specificity:.4f}')

   # Plot ROC curve and calculate AUC
   plot_roc_curve(y_true, y_pred_probs)

# Example usage
data_path = r"D:\Newfolder\s7\project\spectrogram images\using feature slection algorithm\spect 224"
train_and_evaluate(data_path, num_epochs=15)


# In[19]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
   plt.xlabel('Predicted labels')
   plt.ylabel('True labels')
   plt.title('Confusion Matrix')
   plt.show()

def plot_roc_curve(y_true, y_pred_probs):
   fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
   roc_auc = auc(fpr, tpr)

   plt.figure()
   lw = 2
   plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
   plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic (ROC) Curve')
   plt.legend(loc="lower right")
   plt.show()

def train_and_evaluate(data_path, split_ratio=0.8, num_epochs=15):
   # Image dimensions
   height, width = 224, 224  # VGG16 expects input size (224, 224)

   # Data transformation
   transform = transforms.Compose([
       transforms.Resize((height, width)),
       transforms.ToTensor(),
   ])

   # Create dataset
   dataset = ImageFolder(root=data_path, transform=transform)

   # Split dataset into training and validation sets
   total_size = len(dataset)
   train_size = int(split_ratio * total_size)
   val_size = total_size - train_size

   train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

   # Create loaders
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

   # Load pre-trained VGG16 model
   model = vgg16(pretrained=True)
   
   # Modify the last fully connected layer for binary classification
   in_features = model.classifier[-1].in_features
   model.classifier[-1] = nn.Linear(in_features, 2)

   model.train()

   # Loss function and optimizer
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # Lists to store the learning curve data
   train_losses = []
   train_accuracies = []
   val_accuracies = []

   # Training loop
   for epoch in range(num_epochs):
       # Training
       model.train()
       running_loss = 0.0
       correct_train = 0
       total_train = 0
       for inputs, labels in train_loader:
           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           running_loss += loss.item()

           _, predicted = torch.max(outputs.data, 1)
           total_train += labels.size(0)
           correct_train += (predicted == labels).sum().item()

       average_train_loss = running_loss / len(train_loader)
       train_losses.append(average_train_loss)
       train_accuracy = correct_train / total_train
       train_accuracies.append(train_accuracy)

       # Validation
       model.eval()
       correct_val = 0
       total_val = 0
       y_pred_probs = []
       y_true = []
       y_pred = []
       with torch.no_grad():
           for inputs, labels in val_loader:
               outputs = model(inputs)
               _, predicted = torch.max(outputs.data, 1)
               total_val += labels.size(0)
               correct_val += (predicted == labels).sum().item()
               y_true.extend(labels.numpy())
               y_pred.extend(predicted.numpy())
               y_pred_probs.extend(torch.softmax(outputs, 1)[:, 1].tolist())  # Probability for class 1 (positive)

       val_accuracy = correct_val / total_val
       val_accuracies.append(val_accuracy)

       # Print training loss and validation accuracy
       print(f'Epoch {epoch + 1}/{num_epochs}, '
             f'Training Loss: {average_train_loss:.4f}, '
             f'Training Accuracy: {train_accuracy:.4f}, '
             f'Validation Accuracy: {val_accuracy:.4f}')

   # Plot the learning curve
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(train_losses, label='Training Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()

   plt.subplot(1, 2, 2)
   plt.plot(train_accuracies, label='Training Accuracy')
   plt.plot(val_accuracies, label='Validation Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()

   plt.show()

   # Print final training and validation accuracy
   print(f'Final Training Accuracy: {train_accuracies[-1]:.4f}')
   print(f'Final Validation Accuracy: {val_accuracies[-1]:.4f}')

   # Evaluation
   y_true = np.array(y_true)
   y_pred = np.array(y_pred)
   y_pred_probs = np.array(y_pred_probs)

   # Print predicted class values and true class values
   print("Predicted Class Values:", y_pred)
   print("True Class Values:", y_true)

   # Print confusion matrix and classification report
   print(classification_report(y_true, y_pred))
   plot_confusion_matrix(y_true, y_pred)

   # Calculate and print specificity and sensitivity
   tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
   sensitivity = tp / (tp + fn)
   specificity = tn / (tn + fp)
   print(f'Sensitivity: {sensitivity:.4f}')
   print(f'Specificity: {specificity:.4f}')

   # Plot ROC curve and calculate AUC
   plot_roc_curve(y_true, y_pred_probs)

# Example usage
data_path = r"D:\Newfolder\s7\project\spectrogram images\using feature slection algorithm\spect 224"
train_and_evaluate(data_path, num_epochs=15)


# In[1]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import alexnet  # Import AlexNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def train_and_evaluate(data_path, split_ratio=0.8, num_epochs=50):
    # Image dimensions
    height, width = 224, 224

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = ImageFolder(root=data_path, transform=transform)

    # Split dataset into training and validation sets
    total_size = len(dataset)
    train_size = int(split_ratio * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load pre-trained AlexNet model
    model = alexnet(pretrained=True)
    
    # Modify the last fully connected layer for binary classification
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)

    model.train()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Lists to store the learning curve data
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        average_train_loss = running_loss / len(train_loader)
        train_losses.append(average_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Print training loss and validation accuracy
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Training Loss: {average_train_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')

    # Plot the learning curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # Print final training and validation accuracy
    print(f'Final Training Accuracy: {train_accuracies[-1]:.4f}')
    print(f'Final Validation Accuracy: {val_accuracies[-1]:.4f}')

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            predictions = torch.argmax(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Print confusion matrix and classification report
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

# Example usage
data_path = r"D:\Newfolder\s7\project\spectrogram images\using feature slection algorithm\spect 224"
train_and_evaluate(data_path)


# In[ ]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import alexnet  # Import AlexNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def train_and_evaluate(data_path, split_ratio=0.8, num_epochs=50):
   # Image dimensions
   height, width = 224, 224

   # Data transformation
   transform = transforms.Compose([
       transforms.Resize((height, width)),
       transforms.ToTensor(),
   ])

   # Create dataset
   dataset = ImageFolder(root=data_path, transform=transform)

   # Split dataset into training and validation sets
   total_size = len(dataset)
   train_size = int(split_ratio * total_size)
   val_size = total_size - train_size

   train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

   # Create loaders
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

   # Load pre-trained AlexNet model
   model = alexnet(pretrained=True)
   
   # Modify the last fully connected layer for binary classification
   model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)

   model.train()

   # Loss function and optimizer
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # Lists to store the learning curve data
   train_losses = []
   train_accuracies = []
   val_accuracies = []

   # Training loop
   for epoch in range(num_epochs):
       # Training
       model.train()
       running_loss = 0.0
       correct_train = 0
       total_train = 0
       for inputs, labels in train_loader:
           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           running_loss += loss.item()

           _, predicted = torch.max(outputs.data, 1)
           total_train += labels.size(0)
           correct_train += (predicted == labels).sum().item()

       average_train_loss = running_loss / len(train_loader)
       train_losses.append(average_train_loss)
       train_accuracy = correct_train / total_train
       train_accuracies.append(train_accuracy)

       # Validation
       model.eval()
       correct_val = 0
       total_val = 0
       with torch.no_grad():
           for inputs, labels in val_loader:
               outputs = model(inputs)
               _, predicted = torch.max(outputs.data, 1)
               total_val += labels.size(0)
               correct_val += (predicted == labels).sum().item()

       val_accuracy = correct_val / total_val
       val_accuracies.append(val_accuracy)

       # Print training loss and validation accuracy
       print(f'Epoch {epoch + 1}/{num_epochs}, '
             f'Training Loss: {average_train_loss:.4f}, '
             f'Training Accuracy: {train_accuracy:.4f}, '
             f'Validation Accuracy: {val_accuracy:.4f}')

   # Plot the learning curve
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(train_losses, label='Training Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()

   plt.subplot(1, 2, 2)
   plt.plot(train_accuracies, label='Training Accuracy')
   plt.plot(val_accuracies, label='Validation Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()

   plt.show()

   # Print final training and validation accuracy
   print(f'Final Training Accuracy: {train_accuracies[-1]:.4f}')
   print(f'Final Validation Accuracy: {val_accuracies[-1]:.4f}')

   # Evaluation
   model.eval()
   y_true = []
   y_pred = []

   with torch.no_grad():
       for inputs, labels in val_loader:
           outputs = model(inputs)
           predictions = torch.argmax(outputs, 1)

           y_true.extend(labels.numpy())
           y_pred.extend(predictions.numpy())

   y_true = np.array(y_true)
   y_pred = np.array(y_pred)

   # Print confusion matrix and classification report
   print(confusion_matrix(y_true, y_pred))
   print(classification_report(y_true, y_pred))

# Example usage
data_path = r"D:\Newfolder\s7\project\spectrogram images\using paerson coeffiecient fsa\spect 224"
train_and_evaluate(data_path)


# In[3]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
   plt.xlabel('Predicted labels')
   plt.ylabel('True labels')
   plt.title('Confusion Matrix')
   plt.show()

def plot_roc_curve(y_true, y_pred_probs):
   fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
   roc_auc = auc(fpr, tpr)

   plt.figure()
   lw = 2
   plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
   plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic (ROC) Curve')
   plt.legend(loc="lower right")
   plt.show()

def train_and_evaluate(data_path, split_ratio=0.8, num_epochs=15):
   # Image dimensions
   height, width = 224, 224  # VGG16 expects input size (224, 224)

   # Data transformation
   transform = transforms.Compose([
       transforms.Resize((height, width)),
       transforms.ToTensor(),
   ])

   # Create dataset
   dataset = ImageFolder(root=data_path, transform=transform)

   # Split dataset into training and validation sets
   total_size = len(dataset)
   train_size = int(split_ratio * total_size)
   val_size = total_size - train_size

   train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

   # Create loaders
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

   # Load pre-trained VGG16 model
   model = vgg16(pretrained=True)
   
   # Modify the last fully connected layer for binary classification
   in_features = model.classifier[-1].in_features
   model.classifier[-1] = nn.Linear(in_features, 2)

   model.train()

   # Loss function and optimizer
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # Lists to store the learning curve data
   train_losses = []
   train_accuracies = []
   val_accuracies = []

   # Training loop
   for epoch in range(num_epochs):
       # Training
       model.train()
       running_loss = 0.0
       correct_train = 0
       total_train = 0
       for inputs, labels in train_loader:
           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           running_loss += loss.item()

           _, predicted = torch.max(outputs.data, 1)
           total_train += labels.size(0)
           correct_train += (predicted == labels).sum().item()

       average_train_loss = running_loss / len(train_loader)
       train_losses.append(average_train_loss)
       train_accuracy = correct_train / total_train
       train_accuracies.append(train_accuracy)

       # Validation
       model.eval()
       correct_val = 0
       total_val = 0
       y_pred_probs = []
       y_true = []
       y_pred = []
       with torch.no_grad():
           for inputs, labels in val_loader:
               outputs = model(inputs)
               _, predicted = torch.max(outputs.data, 1)
               total_val += labels.size(0)
               correct_val += (predicted == labels).sum().item()
               y_true.extend(labels.numpy())
               y_pred.extend(predicted.numpy())
               y_pred_probs.extend(torch.softmax(outputs, 1)[:, 1].tolist())  # Probability for class 1 (positive)

       val_accuracy = correct_val / total_val
       val_accuracies.append(val_accuracy)

       # Print training loss and validation accuracy
       print(f'Epoch {epoch + 1}/{num_epochs}, '
             f'Training Loss: {average_train_loss:.4f}, '
             f'Training Accuracy: {train_accuracy:.4f}, '
             f'Validation Accuracy: {val_accuracy:.4f}')

   # Plot the learning curve
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(train_losses, label='Training Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()

   plt.subplot(1, 2, 2)
   plt.plot(train_accuracies, label='Training Accuracy')
   plt.plot(val_accuracies, label='Validation Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()

   plt.show()

   # Print final training and validation accuracy
   print(f'Final Training Accuracy: {train_accuracies[-1]:.4f}')
   print(f'Final Validation Accuracy: {val_accuracies[-1]:.4f}')

   # Evaluation
   y_true = np.array(y_true)
   y_pred = np.array(y_pred)
   y_pred_probs = np.array(y_pred_probs)

   # Print predicted class values and true class values
   print("Predicted Class Values:", y_pred)
   print("True Class Values:", y_true)

   # Print confusion matrix and classification report
   print(classification_report(y_true, y_pred))
   plot_confusion_matrix(y_true, y_pred)

   # Calculate and print specificity and sensitivity
   tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
   sensitivity = tp / (tp + fn)
   specificity = tn / (tn + fp)
   print(f'Sensitivity: {sensitivity:.4f}')
   print(f'Specificity: {specificity:.4f}')

   # Plot ROC curve and calculate AUC
   plot_roc_curve(y_true, y_pred_probs)

# Example usage
data_path = r"D:\Newfolder\s7\project\spectrogram images\using paerson coeffiecient fsa\spect 224"
train_and_evaluate(data_path, num_epochs=15)


# In[ ]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
  plt.xlabel('Predicted labels')
  plt.ylabel('True labels')
  plt.title('Confusion Matrix')
  plt.show()

def plot_roc_curve(y_true, y_pred_probs):
  fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
  roc_auc = auc(fpr, tpr)

  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc="lower right")
  plt.show()

def train_and_evaluate(data_path, split_ratio=0.8, num_epochs=15):
  # Image dimensions
  height, width = 224, 224  # VGG16 expects input size (224, 224)

  # Data transformation
  transform = transforms.Compose([
      transforms.Resize((height, width)),
      transforms.ToTensor(),
  ])

  # Create dataset
  dataset = ImageFolder(root=data_path, transform=transform)

  # Split dataset into training and validation sets
  total_size = len(dataset)
  train_size = int(split_ratio * total_size)
  val_size = total_size - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  # Create loaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  # Load pre-trained VGG16 model
  model = vgg16(pretrained=True)
  
  # Modify the last fully connected layer for binary classification
  in_features = model.classifier[-1].in_features
  model.classifier[-1] = nn.Linear(in_features, 2)

  model.train()

  # Loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Lists to store the learning curve data
  train_losses = []
  train_accuracies = []
  val_accuracies = []

  # Training loop
  for epoch in range(num_epochs):
      # Training
      model.train()
      running_loss = 0.0
      correct_train = 0
      total_train = 0
      for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

          _, predicted = torch.max(outputs.data, 1)
          total_train += labels.size(0)
          correct_train += (predicted == labels).sum().item()

      average_train_loss = running_loss / len(train_loader)
      train_losses.append(average_train_loss)
      train_accuracy = correct_train / total_train
      train_accuracies.append(train_accuracy)

      # Validation
      model.eval()
      correct_val = 0
      total_val = 0
      y_pred_probs = []
      y_true = []
      y_pred = []
      with torch.no_grad():
          for inputs, labels in val_loader:
              outputs = model(inputs)
              _, predicted = torch.max(outputs.data, 1)
              total_val += labels.size(0)
              correct_val += (predicted == labels).sum().item()
              y_true.extend(labels.numpy())
              y_pred.extend(predicted.numpy())
              y_pred_probs.extend(torch.softmax(outputs, 1)[:, 1].tolist())  # Probability for class 1 (positive)

      val_accuracy = correct_val / total_val
      val_accuracies.append(val_accuracy)

      # Print training loss and validation accuracy
      print(f'Epoch {epoch + 1}/{num_epochs}, '
            f'Training Loss: {average_train_loss:.4f}, '
            f'Training Accuracy: {train_accuracy:.4f}, '
            f'Validation Accuracy: {val_accuracy:.4f}')

  # Plot the learning curve
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(train_accuracies, label='Training Accuracy')
  plt.plot(val_accuracies, label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.show()

  # Print final training and validation accuracy
  print(f'Final Training Accuracy: {train_accuracies[-1]:.4f}')
  print(f'Final Validation Accuracy: {val_accuracies[-1]:.4f}')

  # Evaluation
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  y_pred_probs = np.array(y_pred_probs)

  # Print predicted class values and true class values
  print("Predicted Class Values:", y_pred)
  print("True Class Values:", y_true)

  # Print confusion matrix and classification report
  print(classification_report(y_true, y_pred))
  plot_confusion_matrix(y_true, y_pred)

  # Calculate and print specificity and sensitivity
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)
  print(f'Sensitivity: {sensitivity:.4f}')
  print(f'Specificity: {specificity:.4f}')

  # Plot ROC curve and calculate AUC
  plot_roc_curve(y_true, y_pred_probs)

# Example usage
data_path = r"D:\Newfolder\s7\project\spectrogram images\using paerson coeffiecient fsa\spect 224"
train_and_evaluate(data_path, num_epochs=15)


# In[ ]:




