import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, ReLU, BatchNormalization, GlobalMaxPool2D
from keras.applications import ResNet152V2
import matplotlib.pyplot as plt
from keras.optimizers import adam



input_path = []
label = []
i = 0
for c in os.listdir("C:/Users/sidst/OneDrive/Desktop/GeoGuesser/newgeoguesser2"):
    temp_path = []
    temp_label = []
    for path in os.listdir("C:/Users/sidst/OneDrive/Desktop/GeoGuesser/newgeoguesser2/"+c):
        temp_label.append(i)
        temp_path.append(os.path.join("C:/Users/sidst/OneDrive/Desktop/GeoGuesser/newgeoguesser2", c, path))
    input_path.append(temp_path)
    label.append(temp_label)
    i = i + 1

data = pd.DataFrame()
data['images'] = input_path[0]
data['label'] = label[0]
data = data.sample(frac=1).astype('str')
train, test = train_test_split(data, test_size=0.1, random_state=20)

for i in range(1, len(label)):
    data = pd.DataFrame()
    data['images'] = input_path[i]
    data['label'] = label[i]
    data = data.sample(frac=1).astype('str')
    temp_train, temp_test = train_test_split(data, test_size=0.1, random_state=20)
    train = pd.concat([train, temp_train], ignore_index=True)
    test = pd.concat([test, temp_test], ignore_index=True)
train = train.sample(frac=1).astype('str')
test = test.sample(frac=1).astype('str')
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_gen = ImageDataGenerator(
    rescale=1./255
)

train_iterator = train_gen.flow_from_dataframe(
    train,
    x_col="images",
    y_col="label",
    target_size=(256, 256),
    batch_size=16,
    class_mode="sparse"
)
test_iterator = test_gen.flow_from_dataframe(
    test,
    x_col="images",
    y_col="label",
    target_size=(256, 256),
    batch_size=16,
    class_mode="sparse"
)
weight_decay = 0.0001
regularizer = tf.keras.regularizers.l2(weight_decay)
model = Sequential([
    Conv2D(32, (5,5), activation='relu', kernel_regularizer=regularizer, input_shape=(256, 256, 3)),
    # BatchNormalization(),
    Conv2D(32, (5,5), activation='relu', kernel_regularizer=regularizer),
    # BatchNormalization(),
    MaxPool2D((2,2)),
    # Dropout(0.2),

    Conv2D(64, (5,5), activation='relu', kernel_regularizer=regularizer),
    # BatchNormalization(),
    Conv2D(64, (5,5), activation='relu', kernel_regularizer=regularizer),
    # BatchNormalization(),
    MaxPool2D((2,2)),
    # Dropout(0.3),
    

    Conv2D(128, (5,5), activation='relu', kernel_regularizer=regularizer),
    BatchNormalization(),
    Conv2D(128, (5,5), activation='relu', kernel_regularizer=regularizer),
    BatchNormalization(),
    MaxPool2D((2,2)),
    # GlobalMaxPool2D(),
    # Dropout(0.3),

    Conv2D(256, (5,5), activation='relu', kernel_regularizer=regularizer),
    BatchNormalization(),
    Conv2D(256, (5,5), activation='relu', kernel_regularizer=regularizer),
    BatchNormalization(),
    # MaxPool2D((2,2)),
    GlobalMaxPool2D(),
    Dropout(0.3),

    # Conv2D(512, (3,3), activation='relu', kernel_regularizer=regularizer),
    # BatchNormalization(),
    # Conv2D(512, (3,3), activation='relu', kernel_regularizer=regularizer),
    # BatchNormalization(),
    # MaxPool2D((2,2)),
    # # GlobalMaxPool2D(),
    # Dropout(0.3),

    # Conv2D(1024, (3,3), activation='relu', kernel_regularizer=regularizer),
    # BatchNormalization(),
    # Conv2D(1024, (3,3), activation='relu', kernel_regularizer=regularizer),
    # BatchNormalization(),
    # MaxPool2D((2,2)),
    # # GlobalMaxPool2D(),
    # Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(102, activation='softmax'),

])

opt = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# model2 = ResNet152V2(include_top=False, input_shape=(256,256,3))
# model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# model2.summary()

history = model.fit(train_iterator, epochs=10, validation_data=test_iterator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()
plt.plot()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.figure()
plt.plot()