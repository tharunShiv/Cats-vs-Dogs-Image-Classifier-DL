'''
 Images will be uploaded soon
'''

# no preprocessing here
# BUT FEATURE SCALING is very important, but later

# Part 1 - Building the CNN

'''
    You must have test folder, train folder
    within each one of them, cats folder & dog folder
    within them we will have cats photos & dog photos
    
'''

# Importing the Keras libraries and packages
from keras.models import Sequential   # initialize the neural network
from keras.layers import Conv2D       # first step of CNN ( 2D because -> images)
from keras.layers import MaxPooling2D  # for pooling
from keras.layers import Flatten    # for flatenning
from keras.layers import Dense   # to add the fully connected layers  

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# conv2D(filters, (kernel_size))
# filters: number of filter  maps -> common practice is to use 32 in the beginning
# and later next layer with 64, 128 , 256 so on..
# but due to CPU constraints

# kernel_size: 3x3 feature size

#inpput_shape: we will convert all of our images into one particular format, since everyone will not be the same
# refer the beginning of CNN Notes, (RGB)
# so for RGB -> (3, 256, 256) -> this format is in theano package
# but due to CPU Contraints
# 64, 64, 3 ->this is in tensorflow backend format

# relu to remove linearity, and to remove the negative values
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# so in another conv layer, no need to add input_shape
# we can also make the 32->64 here
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# pool_size - (2, 2) is recommended

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# units-> number of nodes in this layer -> 128 is decent, we can play with this value
# this is for the ANN I suppose
classifier.add(Dense(units = 128, activation = 'relu'))
# since it is a classification problem wit two classes
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# adam->stochastic gradient descent
# loss>
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# to prevent overfitting
# a great result on training set, poor resut in test set


'''  REfer Keras documentation  '''
# refer notes about data augmentation.

'''
We r working with 10000 images as our data now,
but for the best results we need lakhs of data

how are we gona satisfy that?
using either more data / data augmentation

it will create many batches of images,
creating random transformation of each images

hence it generates moreee images than what exists today..

also prevents overfitting
'''

'
# this is only one of the kind of transformation
# check image preprocessing docs of keras
from keras.preprocessing.image import ImageDataGenerator
# rescale, all values will be between 0 and 1
# 0.2 means, how much we wanna apply them
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
# target size, since we have already chosen 64 64 in our input shape
# batch size after which our weights will be updated
# cats and dogs, so binary
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# higher target size in both results to better results
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# number of images we have in our training set
# since we have 8000 images

# 50 was there in the beginning
# number of images in the test set - 2000
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)


# Part 3 - Making new predictions

'''MAY WORK'''
from skimage.io import imread
from skimage.transform import resize
import numpy as np
class_labels = {v: k for k, v in training_set.class_indices.items()}
 
path_to_file =''
img = imread(path_to_file) #make sure that path_to_file contains the path to the image you want to predict on. 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)
 
if(np.max(img)>1):
    img = img/255.0
 
prediction = classifier.predict_classes(img)
 
print(class_labels[prediction[0][0]])
