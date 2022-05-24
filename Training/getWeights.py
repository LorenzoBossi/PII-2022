import tensorflow as tf
import numpy
import sys
from keras.models import load_model

numpy.set_printoptions(threshold=sys.maxsize)

model = load_model('model.h5')
model.summary()


print("\n\n-------------------- CONVOLUTIONAL LAYER  --------------------\n");

print(model.layers[0].get_weights());



print("\n\n-------------------- DENSE LAYER --------------------\n");

print(model.layers[3].get_weights());




