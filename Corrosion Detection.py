# -*- coding: utf-8 -*-




from gc import callbacks
import sys
import os
import keras
import tensorflow
import numpy as np
import cv2
from tkinter import *

from keras.applications.vgg16 import VGG16
# load model
model = VGG16()
# summarize the model
model.summary()
from tensorflow.keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from time import time
# sys.path.append(r'C:\\Users\\Admin\\Desktop\\Projects\\corossionRed_particles.py')
import Red_particles
con_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))
con_base.summary()


def prototyping():
    prototype = models.Sequential()
    prototype.add(con_base)
    prototype.add(layers.Flatten())
    prototype.add(layers.Dense(256, activation='relu'))
    prototype.add(layers.Dense(1, activation='sigmoid'))
    prototype.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    prototype.summary()
    return prototype

def train_model():
    
    train_loc = 'train'
    validation_loc = 'validation'
    train_dataset = ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
    test_dataset = ImageDataGenerator(rescale=1./255)
    trainner = train_dataset.flow_from_directory(train_loc,target_size=(150, 150),batch_size=4,class_mode='binary')
    validator = test_dataset.flow_from_directory(validation_loc,target_size=(150, 150),batch_size=10,class_mode='binary')
    print(validator)
    return trainner, validator

def main_module():
    model = prototyping()
    trainner, validator = train_model()
    tensor_board = keras.callbacks.TensorBoard(log_dir='output/{}'.format(time()))
    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(learning_rate=2e-5),metrics=['acc'])
    print(trainner)
    print(validator)
    print(callbacks)
    model.fit(trainner,steps_per_epoch=10,epochs=10,validation_data=validator,validation_steps=10,verbose=2,callbacks=[tensor_board])
    model.save('model\\model.h5') 

    
    # image_path = 'C:\\Users\\Admin\\Desktop\\Projects\\corossion\\norust.6.jpg'
    # input_image = image.load_img(image_path, target_size=(150, 150))
    # image_test = image.img_to_array(input_image)
    # image_test = image_test.reshape((1,) + image_test.shape)
    # image_test =image_test.astype('float32') / 255
    # rust_prob = model.predict(image_test)
    # #print(rust_prob)
    # if (rust_prob > 0.50):
    #     result = "This is a Rust image"
        
    #     depth = 15
    #     thresh_hold = 0.8
    #     distance = 5
    #     thresh = 0.07
    #     img = Red_particles.scale_image(image_path,max_size=1000000)
    #     Red_particles.energy_gLCM(img,depth,thresh_hold,distance,thresh)
    # else:
    #     result = "This is a no Rust image"
        
    # ws = Tk()
    # ws.title('PythonGuides')
    # ws.geometry('400x300')
    # ws.config(bg='#84BF04')

    # message = result

    # text_box = Text(
    #     ws,
    #     height=12,
    #     width=40
    # )
    # text_box.pack(expand=True)
    # text_box.insert('end', message)
    # text_box.config(state='disabled')

    # ws.mainloop()   
        
    
    return

main_module()