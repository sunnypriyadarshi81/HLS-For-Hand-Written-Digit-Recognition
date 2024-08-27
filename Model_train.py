import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# from keras2c import k2c

(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
y_train = keras.utils.to_categorical(y_train, 10)

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),

    layers.Dense(10, activation="softmax")])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)

model.save('model.h5')