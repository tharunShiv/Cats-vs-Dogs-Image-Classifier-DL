
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

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
