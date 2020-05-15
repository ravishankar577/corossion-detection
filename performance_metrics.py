

import keras
import numpy as np
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

model = keras.models.load_model('model\\model.h5')

validation_loc = 'validation'
test_dataset = ImageDataGenerator(rescale=1./255)
validator = test_dataset.flow_from_directory(validation_loc,target_size=(150, 150),batch_size=10,class_mode='binary')
filenames = validator.filenames
nb_samples = len(filenames)
predictions=model.predict(validator, steps=nb_samples)
val_preds = np.zeros((predictions.shape[0],1))
for i in range(predictions.shape[0]):
    if predictions[i]>0.5:
        val_preds[i] = 1
    else:
        val_preds[i] = 0
val_trues = validator.classes
labels = validator.class_indices.keys()
from sklearn.metrics import classification_report
report = classification_report(val_trues, val_preds, target_names=labels)
print(report)