# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 01:43:07 2020

@author: devel
"""
import os
os.chdir("M:/Udemy/deep_learning")

#%%
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv", sep = ",")

#%%

x_data = data.drop(["id","acmotor"],axis=1)

x_data.bearings = [0 if each == "Ok" else 1 for each in data.bearings]
x_data.wpump = [0 if each == "Ok" else 1 for each in data.wpump]
x_data.radiator = [0 if each == "Clean" else 1 for each in data.radiator]
x_data.exvalve = [0 if each == "Clean" else 1 for each in data.exvalve]

# 0 = Ok, 1 = Noisy for bearings abd water pump
# 0 = Clean, 1 = Dirty for radiator and exhaust valve

#%%

maxval = np.max(x_data)
minval = np.min(x_data)
std_dv = np.std(x_data)
mean = np.mean(x_data)

print(maxval,minval)

#%%

splitted_data = np.split(x_data, [20], axis=1)
x_data = splitted_data[0]
y_data = splitted_data[1]

#%%

col_names = x_data.columns.values.tolist()

norm_minmaxdata = {'minnorm': [0,0,0,0,0,30,60,0,30,30,200,30,250,40,0,0,0,0,0,0], 'maxnorm': [3000,25000,150,12,1800,100,200,6,150,200,275,80,320,50,2,2,12,2,2,10]}
norm_minmax = pd.DataFrame(norm_minmaxdata, columns = ['minnorm','maxnorm'])

minnorm = norm_minmax.minnorm.values
maxnorm = norm_minmax.maxnorm.values

#%%

x_norm = (x_data - minnorm)/(maxnorm-minnorm)

#%%

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, Flatten
from keras.optimizers import SGD

#%%

model = Sequential()
model.add(Dense(32,input_dim=20))
model.add(Activation('relu'))

model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(4))
model.add(Activation('relu'))
model.summary()

#%%

model.compile(optimizer = "rmsprop", loss = 'mse',metrics=['accuracy'])

#%%

history = model.fit(x_norm,y_data,epochs=100,validation_split=0.2)

#%%
from keras.models import model_from_json
from keras.models import load_model


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")


model.save('model.hdf5')
loaded_model=load_model('model.hdf5')

#%%

score = model.evaluate(x_norm, y_data,verbose=0)

print('Test Loss:', score[0])
print('Test accuracy:', score[1])

#%%

y_test_pred_class = model.predict_classes(x_norm, verbose = 1)
y_test_pred = model.predict(x_norm, verbose = 1)

#%%

import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()