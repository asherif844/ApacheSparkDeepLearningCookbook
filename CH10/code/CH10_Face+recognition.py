
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten 
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


pwd


# In[3]:


cd desktop


# In[4]:


#reading images from the local drive 
mypath='MIT-CBCL-facerec-database//training-synthetic' 
onlyfiles= [ f for f in listdir(mypath) if isfile(join(mypath,f)) ] 
images =np.empty([3240,200,200],dtype=int)
for n in range(0, len(onlyfiles)):
 images[n] = mpimg.imread( join(mypath,onlyfiles[n]) ).astype(np.float32)


# In[5]:


plt.imshow (images[0])


# In[6]:


plt.imshow (images[1])


# In[7]:


plt. imshow (images[2])


# In[8]:


plt.imshow(images[3119])


# In[9]:


y =np.empty([3240,1],dtype=int) 
for x in range(0, len(onlyfiles)):
    if onlyfiles[x][3]=='0': y[x]=0
    elif onlyfiles[x][3]=='1': y[x]=1
    elif onlyfiles[x][3]=='2': y[x]=2
    elif onlyfiles[x][3]=='3': y[x]=3
    elif onlyfiles[x][3]=='4': y[x]=4
    elif onlyfiles[x][3]=='5': y[x]=5
    elif onlyfiles[x][3]=='6': y[x]=6
    elif onlyfiles[x][3]=='7': y[x]=7
    elif onlyfiles[x][3]=='8': y[x]=8
    elif onlyfiles[x][3]=='9': y[x]=9


# In[10]:


#funtion for cropping images to obtain only the significant part 
def crop(img):
    a=28*np.ones(len(img)) #background has pixel intensity of 28 
    b=np.where((img== a).all(axis=1)) #check image background
    img=np.delete(img,(b),0) #deleting the unwanted part from the Y axis 
    plt.imshow(img)
    img=img.transpose()
    d=28*np.ones(len(img[0]))
    e=np.where((img== d).all(axis=1))
    img=np.delete(img,e,0) #deleting the unwanted part from the X axis 
    img=img.transpose()
    print (img.shape) #printing image shape to ensure it is actually being cropped
    super_threshold_indices = img < 29 #padding zeros instead of background data  
    img[super_threshold_indices] = 0
    plt.imshow (img)
    return img[0:150, 0:128]


# In[11]:


#cropping all the images
image = np.empty([3240,150,128],dtype=int) 
for n in range(0, len(images)):
 image[n]=crop(images[n])


# In[12]:


print (image[22])


# In[13]:


print (image[22].shape)


# In[14]:


# randomly splitting data into training(80%) and test(20%) sets 
test_ind=np.random.choice(range(3240), 648, replace=False) 
train_ind=np.delete(range(0,len(onlyfiles)),test_ind)


# In[15]:


# segregating the training and test images 
x_train=image[train_ind] 
y1_train=y[train_ind] 
x_test=image[test_ind] 
y1_test=y[test_ind]


# In[16]:


#reshaping the input images
x_train = x_train.reshape(x_train.shape[0], 128, 150, 1) 
x_test = x_test.reshape(x_test.shape[0], 128, 150, 1)


# In[17]:


#converting data to float32
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')


# In[18]:


#normalizing data
x_train/=255 
x_test/=255
#10 digits represent the 10 classes 
number_of_persons = 10


# In[19]:


#convert data to vectors
y_train = np_utils.to_categorical(y1_train, number_of_persons) 
y_test = np_utils.to_categorical(y1_test, number_of_persons)


# In[25]:


# model building
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(128,150,1))) #Input layer 
model.add(Activation('relu')) # 'relu' as activation function
model.add(Conv2D(16, (3, 3))) #first hidden layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # Maxpooling from (2,2)
model.add(Conv2D(16,(3, 3))) # second hidden layer 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # Maxpooling from (2,2)
model.add(Flatten()) #flatten the maxpooled data
# Fully connected layer
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25)) #Dropout is applied to overcome overfitting 
model.add(Dense(10)) 
#output layer
model.add(Activation('softmax')) # 'softmax' is used for SGD


# In[26]:


model.summary()


# In[27]:


#model compliation
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[28]:


# data augmentation to reduce overfitting problem
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, 
                         height_shift_range=0.08,zoom_range=0.08)
test_gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=16) 
test_generator = test_gen.flow(x_test, y_test, batch_size=16)


# In[29]:


#model fitting
model.fit_generator(train_generator, epochs=5, validation_data=test_generator) 
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0) 
print("Recognition Error: %.2f%%" % (100-scores[1]*100))

