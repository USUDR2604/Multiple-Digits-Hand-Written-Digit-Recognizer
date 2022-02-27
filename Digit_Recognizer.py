#importing libraries
import tensorflow as tf  
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers
from keras import regularizers
from keras import models
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
#from tensorflow.python.keras import regularizers


#loading data
train=pd.read_csv('DataFiles/train.csv')  #42000images
test=pd.read_csv('DataFiles/test.csv')   #18000images

train_images=train.iloc[:,1:].values
train_labels=train.iloc[:,0:1].values
test_X=test.iloc[:,:].values

sns.countplot(train['label'])
plt.show()

print(train_images)

#data reshaping
train_images = train_images.reshape((-1, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test = test.values.reshape(-1,28,28,1)
test = test / 255.0
print(train_images.shape)

g=plt.imshow(train_images[3][:,:,0], cmap='gray')
g=plt.title(train_labels[3])
plt.show()
og=train_images[4][:,:,0]
plt.imshow(og, cmap='gray')
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_ubyte

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa

footprint = disk(1)
eroded = erosion(og, footprint)
plot_comparison(og, eroded, 'erosion')  

dilated = dilation(og, footprint)
plot_comparison(og, dilated, 'dilation') 
    
opened = opening(og, footprint)
plot_comparison(og, opened, 'opening')    

phantom = og.copy()
phantom[10:30, 200:210] = 0

closed = closing(phantom, footprint)
plot_comparison(phantom, closed, 'closing')    

phantom = og.copy()
phantom[340:350, 200:210] = 255
phantom[100:110, 200:210] = 0

w_tophat = white_tophat(phantom, footprint)
plot_comparison(phantom, w_tophat, 'white tophat')

b_tophat = black_tophat(phantom, footprint)
plot_comparison(phantom, b_tophat, 'black tophat')

sk = skeletonize(og == 0)
plot_comparison(og, sk, 'skeletonize')

hull1 = convex_hull_image(og == 0)
plot_comparison(og, hull1, 'convex hull')

og_mask = og == 0
og_mask[45:50, 75:80] = 1

hull2 = convex_hull_image(og_mask)
plot_comparison(og_mask, hull2, 'convex hull')
plt.show()

train_labels = tf.keras.utils.to_categorical(train_labels)

#using data agumenteation
datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,
        samplewise_std_normalization=False, 
        zca_whitening=False, 
        rotation_range=10, 
        zoom_range = 0.1,  
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=False, 
        vertical_flip=False) 
datagen.fit(train_images)

model = models.Sequential()

model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))
model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.35))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

# Fit the model
history = model.fit_generator(datagen.flow(train_images,train_labels, batch_size=120),
                              epochs = 10, 
                              verbose = 1, steps_per_epoch=train_images.shape[0] // 120
                              )

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='r', label="Training accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()
# predict results
results = model.predict(test)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
#concatinating result with series of numbers from 1 to 28000
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)

model.save('model.h5')