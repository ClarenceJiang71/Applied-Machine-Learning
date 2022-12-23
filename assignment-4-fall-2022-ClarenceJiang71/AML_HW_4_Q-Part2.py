#!/usr/bin/env python
# coding: utf-8

# **Homework 4 Spring 202**
# 
# **Due Date** - **11/23/2022**
# 
# Your Name - Clarence Jiang
# 
# Your UNI - yj2737
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pprint 
pp = pprint.PrettyPrinter(indent=4)
import warnings
warnings.filterwarnings("ignore")


# # PART 2 CIFAR 10 Dataset

# CIFAR-10 is a dataset of 60,000 color images (32 by 32 resolution) across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The train/test split is 50k/10k.

# In[2]:


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.datasets import cifar10
(x_dev, y_dev), (x_test, y_test) = cifar10.load_data()


# In[3]:


LABELS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# In[4]:


y_dev_flatten = y_dev.flatten()


# In[5]:


y_dev_flatten.shape


# 2.1 Plot 5 samples from each class/label from train set on a 10*5 subplot

# In[6]:


#Your code here

import random
fig, axs = plt.subplots(10, 5 , figsize = (20,20))
fig.tight_layout(pad=5.0)
for i in range(10):
    interest_index_list = np.where(y_dev_flatten == i)[0]
    interest_index = random.choices(list(interest_index_list), k=5)
    x_dev_interest = x_dev[interest_index]
    y_dev_interest = y_dev[interest_index]

    for j in range(5):
        image = x_dev_interest[j]
        axs[i,j].imshow(image)
        axs[i,j].set_title(LABELS[i])


# 2.2  Preparing the dataset for CNN 
# 
# 1) Print the shapes - $x_{dev}, y_{dev},x_{test},y_{test}$
# 
# 2) Flatten the images into one-dimensional vectors and again print the shapes of $x_{dev}$,$x_{test}$
# 
# 3) Standardize the development and test sets.
# 
# 4) Train-test split your development set into train and validation sets (8:2 ratio).

# In[7]:


#Your code here
print(f"The shape of X_dev is {x_dev.shape}")
print(f"The shape of y_dev is {y_dev.shape}")
print(f"The shape of X_test is {x_test.shape}")
print(f"The shape of y_test is {y_test.shape}")


# In[8]:


x_dev_rs = x_dev.reshape(50000, 32*32*3)
x_test_rs = x_test.reshape(10000, 32*32*3)

print(f"The shape of X_dev is {x_dev_rs.shape}")
print(f"The shape of X_test is {x_test_rs.shape}")


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_dev_scale = scaler.fit_transform(x_dev_rs)
x_test_scale = scaler.fit_transform(x_test_rs)


# In[10]:


from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
y_dev_cat = to_categorical(y_dev, 10)
x_train, x_val,y_train, y_val = train_test_split(
    x_dev_scale, y_dev_cat, test_size=0.2, random_state=42)


# 2.3 Build the feed forward network 
# 
# First hidden layer size - 128
# 
# Second hidden layer size - 64
# 
# Third and last layer size - You should know this
# 

# In[11]:


#Your code here
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(128, input_shape=(3072,), activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))


# 2.4) Print out the model summary. Can show show the calculation for each layer for estimating the number of parameters

# In[12]:


#Your code here
model.summary()


# 2.5) Do you think this number is dependent on the image height and width? 

# **ans: No I do not think it depends on the image height and width. It more depends on the number of neurons at current and previous layers.**

# **Printing out your model's output on first train sample. This will confirm if your dimensions are correctly set up. The sum of this output equal to 1 upto two decimal places?**

# In[13]:


#modify name of X_train based on your requirement

model.compile()
output = model.predict(x_train[0].reshape(1,-1))

# print(output)
print("Output: {:.2f}".format(sum(output[0])))


# 2.6) Using the right metric and  the right loss function, with Adam as the optimizer, train your model for 20 epochs with batch size 128.

# In[14]:


#Your code here
model.compile("adam", "categorical_crossentropy", metrics = ["accuracy"])

history = model.fit(x_dev_scale, y_dev_cat, batch_size = 128, epochs = 20, validation_split = .2)


# 2.7) Plot a separate plots for:
# 
# a. displaying train vs validation loss over each epoch
# 
# b. displaying train vs validation accuracy over each epoch 

# In[15]:


#Your code here
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[16]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# 2.8) Finally, report the metric chosen on test set.

# In[17]:


#Your code here
y_test_cat = to_categorical(y_test, 10)
score = model.evaluate(x_test_scale, y_test_cat, verbose=0)
print(score)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")


# 2.9 If the accuracy achieved is quite less(<50%), try improve the accuracy [Open ended question, you may try different approaches]

# In[52]:


#Your code here
from tensorflow.keras.layers import Dropout
model2 = Sequential()
model2.add(Dense(128, input_shape=(3072,), activation="relu"))
model2.add(Dense(64, activation="relu"))
model2.add(Dense(32, activation="relu"))
model2.add(Dense(16, activation="relu"))
model2.add(Dense(10, activation="softmax"))


# In[53]:


model2.compile("adam", "categorical_crossentropy", metrics = ["accuracy"])

history2 = model2.fit(x_dev_scale, y_dev_cat, batch_size = 32, epochs = 13, validation_split = .2)


# In[55]:


score2 = model2.evaluate(x_test_scale, y_test_cat, verbose=0)
print(f"Test loss: {score2[0]}")
print(f"Test accuracy: {score2[1]}")


# 2.10 Plot the first 50 samples of test dataset on a 10*5 subplot and this time label the images with both the ground truth (GT) and predicted class (P). (Make sure you predict the class with the improved model)

# In[21]:


#Your code here
y_prob = model.predict(x_test_scale[:50])
y_classes = y_prob.argmax(axis=-1)
y_classes


# In[22]:


rows = 10
columns = 5
fig = plt.figure(figsize = (20, 20))

for i in range(1, rows*columns+1):
    fig.tight_layout(pad=5.0)
    fig.add_subplot(rows, columns, i).set_title(f"Ground: {LABELS[y_test[i-1][0]]}, Predict: {LABELS[y_classes[i-1]]}")
    image = x_test[i-1]
    plt.imshow(image)

        


# # PART 3 Convolutional Neural Network

# In this part of the homework, we will build and train a classical convolutional neural network on the CIFAR Dataset

# In[23]:


from tensorflow.keras.datasets import cifar10
(x_dev, y_dev), (x_test, y_test) = cifar10.load_data()
print("x_dev: {},y_dev: {},x_test: {},y_test: {}".format(x_dev.shape, y_dev.shape, x_test.shape, y_test.shape))

x_dev, x_test = x_dev.astype('float32'), x_test.astype('float32')
x_dev = x_dev/255.0
x_test = x_test/255.0


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(x_dev, y_dev,test_size = 0.2, random_state = 42)


# 3.1 We will be implementing the one of the first CNN models put forward by Yann LeCunn, which is commonly refered to as LeNet-5. The network has the following layers:
# 
# 1) 2D convolutional layer with 6 filters, 5x5 kernel, stride of 1 padded to yield the same size as input, ReLU activation
# 
# 2) Maxpooling layer of 2x2
# 
# 3) 2D convolutional layer with 16 filters, 5x5 kernel, 0 padding, ReLU activation
# 
# 4 )Maxpooling layer of 2x2
# 
# 5) 2D convolutional layer with 120 filters, 5x5 kernel, ReLU activation. Note that this layer has 120 output channels (filters), and each channel has only 1 number. The output of this layer is just a vector with 120 units!
# 
# 6) A fully connected layer with 84 units, ReLU activation
# 
# 7) The output layer where each unit respresents the probability of image being in that category. What activation function should you use in this layer? (You should know this)
# 

# In[40]:


# your code here

# cnn initialize 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
cnn = Sequential()

#1
cnn.add(Conv2D(6, kernel_size = (5,5), activation='relu', input_shape=(32,32,3), padding = "same", strides = 1))
#2
cnn.add(MaxPooling2D(pool_size = (2,2)))
#3
cnn.add(Conv2D(16, kernel_size = (5,5), activation='relu', padding = "valid"))
#4
cnn.add(MaxPooling2D(pool_size = (2,2)))
#5
cnn.add(Conv2D(120, kernel_size = (5,5), activation='relu'))
# 5.5
cnn.add(Flatten())
#6
cnn.add(Dense(84, activation = 'relu'))
#7
cnn.add(Dense(10, activation = 'softmax'))


# 3.2 Report the model summary 

# In[41]:


#your code here
cnn.summary()


# 3.3 Model Training
# 
# 1) Train the model for 20 epochs. In each epoch, record the loss and metric (chosen in part 3) scores for both train and validation sets.
# 
# 2) Plot a separate plots for:
# 
# * displaying train vs validation loss over each epoch
# * displaying train vs validation accuracy over each epoch
# 
# 3) Report the model performance on the test set. Feel free to tune the hyperparameters such as batch size and optimizers to achieve better performance.

# In[42]:


x_dev.reshape(x_dev.shape[0], 32, 32, 3)
x_dev.shape


# In[43]:


# Your code here
cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn = cnn.fit(x_dev, to_categorical(y_dev, 10), batch_size = 128, epochs = 20, validation_split = .1)


# In[44]:


plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[45]:


plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[39]:


y_test_cat = to_categorical(y_test, 10)
score = cnn.evaluate(x_test, y_test_cat, verbose=0)
print(score)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")


# 3.4 Overfitting
# 
# 1) To overcome overfitting, we will train the network again with dropout this time. For hidden layers use dropout probability of 0.3. Train the model again for 20 epochs. Report model performance on test set. 
# 
# Plot a separate plots for:
# 
# *   displaying train vs validation loss over each epoch
# *   displaying train vs validation accuracy over each epoch 
# 
# 2) This time, let's apply a batch normalization after every hidden layer, train the model for 20 epochs, report model performance on test set as above. 
# 
# Plot a separate plots for:
# 
# *   displaying train vs validation loss over each epoch
# *   displaying train vs validation accuracy over each epoch 
# 
# 3) Compare batch normalization technique with the original model and with dropout, which technique do you think helps with overfitting better?

# In[56]:


# Your code here

# 1）
cnn2 = Sequential()
cnn2.add(Conv2D(6, kernel_size = (5,5), activation='relu', input_shape=(32,32,3), padding = "same", strides = 1))
cnn2.add(MaxPooling2D(pool_size = (2,2)))
cnn2.add(Dropout(0.3))
cnn2.add(Conv2D(16, kernel_size = (5,5), activation='relu', padding = "valid"))
cnn2.add(MaxPooling2D(pool_size = (2,2)))
cnn2.add(Dropout(0.3))
cnn2.add(Conv2D(120, kernel_size = (5,5), activation='relu'))
cnn2.add(Flatten())
cnn2.add(Dense(84, activation = 'relu'))
#7
cnn2.add(Dense(10, activation = 'softmax'))


# In[57]:


cnn2.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn2 = cnn2.fit(x_dev, to_categorical(y_dev, 10), batch_size = 128, epochs = 20, validation_split = .2)


# In[58]:


plt.plot(history_cnn2.history['loss'])
plt.plot(history_cnn2.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[59]:


plt.plot(history_cnn2.history['accuracy'])
plt.plot(history_cnn2.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[60]:


score = cnn2.evaluate(x_test, y_test_cat, verbose=0)
print(score)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")


# In[62]:


from tensorflow.keras.layers import BatchNormalization
cnn3 = Sequential()
cnn3.add(Conv2D(6, kernel_size = (5,5), activation='relu', input_shape=(32,32,3), padding = "same", strides = 1))
cnn3.add(MaxPooling2D(pool_size = (2,2)))
cnn3.add(BatchNormalization())
cnn3.add(Conv2D(16, kernel_size = (5,5), activation='relu', padding = "valid"))
cnn3.add(MaxPooling2D(pool_size = (2,2)))
cnn3.add(BatchNormalization())
cnn3.add(Conv2D(120, kernel_size = (5,5), activation='relu'))
cnn3.add(Flatten())
cnn3.add(Dense(84, activation = 'relu'))
cnn3.add(Dense(10, activation = 'softmax'))


# In[63]:


cnn3.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn3 = cnn3.fit(x_dev, to_categorical(y_dev, 10), batch_size = 128, epochs = 20, validation_split = .2)


# In[64]:


plt.plot(history_cnn3.history['loss'])
plt.plot(history_cnn3.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[65]:


plt.plot(history_cnn3.history['accuracy'])
plt.plot(history_cnn3.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[66]:


score = cnn3.evaluate(x_test, y_test_cat, verbose=0)
print(score)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")


# In[ ]:




