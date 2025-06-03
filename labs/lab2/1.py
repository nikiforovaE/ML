import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_0 = pd.read_csv("src/nn_0.csv")
data_1 = pd.read_csv("src/nn_1.csv")

data = pd.concat([data_0, data_1], ignore_index=True)

X = data[['X1', 'X2']].values
y = data['class'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_sigmoid_adam = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

model_sigmoid_adam.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

history_sigmoid_adam = model_sigmoid_adam.fit(X_train, y_train, epochs=50, batch_size=32,
                                              validation_data=(X_test, y_test))

plt.plot(history_sigmoid_adam.history['accuracy'], label='Training Accuracy')
plt.plot(history_sigmoid_adam.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy (Sigmoid, Adam)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
