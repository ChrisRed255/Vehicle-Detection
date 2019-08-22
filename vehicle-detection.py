# -*- coding: utf-8 -*-
"""Vehicle detection.ipynb

# **Mounting Drive to Colab Workspace**

**Mount drive on colab**
"""

from google.colab import drive
drive.mount('/content/gdrive')

"""# -------------------------------------------------------------------------

# Download file from colab to local disk
"""

from google.colab import files

files.download('example.txt')

"""# Preparing dataset"""

!pip install -U -q kaggle
!mkdir -p ~/.kaggle

from google.colab import files
files.upload()

!cp kaggle.json ~/.kaggle/

"""**Kaggle datasets**"""

!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets list

"""**Download Kaggle Cars dataset**"""

!kaggle datasets download -d jessicali9530/stanford-cars-dataset

"""**Copying car dataset to drive**"""

!cp /content/stanford-cars-dataset.zip /content/gdrive/'My Drive'/Colab/

"""**Download the landscape dataset (small dataset)**"""

!wget https://github.com/ml5js/ml5-data-and-models/raw/master/datasets/images/landscapes/landscapes_small.zip

"""# -------------------------------------------------------------------------

# **Use a ready dataset and test dataset**

**Create a workspace folder and copy the dataset to it**
"""

!mkdir Workspace
!cp /content/gdrive/'My Drive'/Colab/training.zip /content/Workspace

cd Workspace

"""**Copy pictures for test**"""

!mkdir test
!cp /content/gdrive/'My Drive'/Colab/pic1.jpg /content/gdrive/'My Drive'/Colab/pic2.jpeg /content/Workspace/test
!cp /content/gdrive/'My Drive'/Colab/vehicle.jpg /content/gdrive/'My Drive'/Colab/vehicle2.jpg /content/Workspace/test
!cp /content/gdrive/'My Drive'/share2.jpg /content/Workspace/test

!cp /content/Workspace/augmentated_images.zip /content/gdrive/'My Drive'/Colab

"""# -------------------------------------------------------------------------

# **Preparing workspace for ready dataset**

**Access "Workspace" folder**
"""

cd Workspace

"""**Unzip training zip file**"""

!unzip training.zip

"""# -------------------------------------------------------------------------

# **Preparing workspace**

**Creating the file "Workspace" and move the zip file to that file**
"""

!mkdir Workspace
!mv stanford-cars-dataset.zip Workspace

"""**Access to the "Workspace" file**"""

cd Workspace

"""**Unzip the dataset file**"""

!unzip stanford-cars-dataset.zip

"""**Create "Car" file that holds the training car images**"""

!mkdir Car
!mv cars_train.zip Car

"""**Access to "Car" file**"""

cd Car

"""**Unzip the training data**"""

!unzip cars_train.zip

"""**Cleaning the training data folder**"""

rm -r __MACOSX/

cd cars_train/

"""**Moving the training data to "Car" folder**"""

!mv *.jpg /content/Workspace/Car

"""**Remove the useless files **"""

rm -r cars_train.zip cars_train/

"""**Creating "Landscape" file and move landscape dataset to it**"""

!mkdir Landscape
!mv landscapes_small.zip Landscape

"""**Access to "Landscape" folder**"""

cd Landscape/

"""**Unzip the landscape dataset**"""

!unzip landscapes_small.zip

"""**Remove useless files**"""

rm -r __MACOSX/ landscapes_small.zip
rm -r city field forest lake mountain ocean road

"""**Create "AllPic" file and move all landscape pics to it**"""

!mkdir AllPic
!mv city/*.jpg forest/*.jpg lake/*.jpg ocean/*.jpg road/*.jpg field/*.jpg mountain/*.jpg AllPic

"""**Move from "AllPic" file to "Landscape" file**"""

!mv AllPic/*.jpg /content/Workspace/Landscape

"""**Return to main folder**"""

cd /content

"""**Remove uiseless files**"""

rm -r cars_annos.mat stanford-cars-dataset.zip

"""**Create "CarTest" folder and move the test file to it **"""

!mkdir CarTest
!mv cars_test.zip CarTest

"""**Unzip the test file**"""

!unzip cars_test.zip

"""**Copy from drive car and landscape pics to test**"""

!cp /content/gdrive/'My Drive'/Colab/pic1.jpg /content/gdrive/'My Drive'/Colab/pic2.jpeg /content/Workspace
!cp /content/gdrive/'My Drive'/Colab/vehicle.jpg /content/gdrive/'My Drive'/Colab/vehicle2.jpg /content/Workspace
!cp /content/gdrive/'My Drive'/share2.jpg /content/Workspace

"""# -------------------------------------------------------------------------

# Importing ready Workspace

**Copy workspace from drive**
"""

!cp /content/gdrive/'My Drive'/Colab/Workspace.zip /content/

"""**Unzip workspace**"""

!unzip Workspace.zip

"""**Delete images folder**"""

rm -r Workspace/images

"""# -------------------------------------------------------------------------

# Using the VGG16 model ready on my drive

**Copy the VGG16 model from drive to workspace**
"""

!cp /content/gdrive/'My Drive'/Colab/vgg16_model.h5 /content/

"""# -------------------------------------------------------------------------

# OpenCV
"""

from google.colab.patches import cv2_imshow

import cv2
import os

img = cv2.imread(os.path.join(train_path,"Car/00001.jpg"), cv2.IMREAD_UNCHANGED)
cv2_imshow(img)

"""# -------------------------------------------------------------------------

# Programming part

**Access workspace**
"""

cd Workspace

"""**Import packages**"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import keras
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

"""**Prepare data for model**"""

DATADIR = '/content/Workspace/train'
CATEGORY = ['Car','Landscape']
IMG_SIZE = 300
training_data =[]

def create_training_data():
  print("Processing ...\n\n")
  for category in CATEGORY:
    path = os.path.join(DATADIR,category)
    class_num = CATEGORY.index(category)
    for img in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        #img_array = img_array / 255
        new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        training_data.append([new_array,class_num])
      except Exception as e:
        print("because of the category :"+category+", file : "+img+"\n"+str(e))
  print("Processing complete !\n\n")

create_training_data()

"""**Shuffle data**"""

for i in range(0,6):
  random.shuffle(training_data)

"""**Create matrix X that holds features and matrix Y that holds labels**"""

X = []
y = []

for features,label in training_data:
  X.append(features)
  y.append(label)

"""**Reshape the features matrix**"""

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

"""**Save X and Y into pickle file to use it after**"""

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

"""**Load pickle data X and Y**"""

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

"""**Normalize data**"""

#X = keras.utils.normalize(X, axis=-1, order=2)
X = X/255.0

model = tf.keras.models.load_model()

"""**Create the model**"""

model = Sequential()
try:
  model.add(Conv2D(32,(3,3),input_shape = X.shape[1:]))
  model.add(Activation("relu"))
  model.add(Conv2D(32,(3,3)))
  model.add(Activation("relu"))
  
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(32,(3,3)))
  model.add(Activation("relu"))
  model.add(Conv2D(32,(3,3)))
  model.add(Activation("relu"))
  
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(32,(3,3)))
  model.add(Activation("relu"))
  model.add(Conv2D(32,(5,5)))
  model.add(Activation("relu"))
  
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))

  model.add(Flatten())
  model.add(Dense(64))
  model.add(Dropout(0.3))
  
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

except Exception as e:
  print("Error :",str(e))

"""**Compile the model**"""

model.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

"""**Train the model**"""

model.fit(X,y,epochs=10,validation_split=0.1)

"""**Save the model**"""

model.save('my_model_16_layers.h5')

"""**Load the model**"""

model = tf.keras.models.load_model("my_model_16_layers.h5")

"""**Test prediction of the model**"""

CATEGORIES = ['Car','Landscape']
directory = '/content/Workspace/CarTest/cars_test/00236.jpg'
def prepare(filepath):
  IMG_SIZE = 300
  img_array= cv2.imread(filepath)
  new_array= cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
  return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

img= plt.imread(directory)
plt.imshow(img)
plt.show()

prediction= model.predict([prepare(directory)])
print("It's a :",CATEGORIES[int(prediction[0][0])])

"""# -------------------------------------------------------------------------

"""

"""**Test prediction of the model**"""

CATEGORIES = ['Car','Landscape']
directory = '/content/Workspace/pic1.jpg'
def prepare(filepath):
  IMG_SIZE = 400
  img_array= cv2.imread(filepath)
  new_array= cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
  return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

img= plt.imread(directory)
plt.imshow(img)
plt.show()

prediction= model.predict([prepare(directory)])
print("It's a :",CATEGORIES[int(prediction[0][0])])

"""# -------------------------------------------------------------------------

"""