from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
import sys, os
import numpy as np
import helpers

# Get the size of the pixels
X_PIXELS = 64
Y_PIXELS = 32
DEPTH = 15

# Define the neural net parameters
random_state = 2022
l2_regularisation = 0.055
learning_rate = 0.0005
epochs = 25
batch_size = 32
test_size = 0.1
valid_size = 0.2

# Read in the data
features_filename = "yardagePredictFeaturesFootball_Lag.pickle"
with open(os.path.join('..', 'processedData', features_filename), "rb") as f:
    data = pickle.load(f)

count = []
for uniqueId in data.keys():
    count.append(len(data[uniqueId]))
from collections import Counter
Counter(count)

y, X = [], []
for uniqueId in data.keys():
    frames = []
    for frame in data[uniqueId]:
        frames.append(np.stack([frame[0], frame[1], frame[2]], axis=2))
    # TODO figure out why ~3% of frames don't have all 5? Counter({5: 2659, 3: 35, 4: 24, 2: 18, 1: 5})
    if len(frames) == 5:
        X.append(np.stack(frames, axis=3).reshape(64, 32, 15))
        y.append(data[uniqueId][0][3])

y = np.array(y)
X = np.array(X)

# Drop any nans in the y value
args = np.argwhere(np.isnan(y))
X = np.delete(X, args, axis=0)
y = np.delete(y, args)

# Split the data
X_trainValid, X_test, y_trainValid, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainValid, y_trainValid, test_size=valid_size, random_state=random_state)

(X_train, y_train) = helpers.dataAugment(X_train, y_train)

# Set up the tensorflow model
model = keras.models.Sequential([
    keras.layers.Conv2D(16, kernel_size=7, activation="relu", padding="same",
                        input_shape=[X_PIXELS, Y_PIXELS, DEPTH]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="relu")
])

# Compile neural network
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss="MeanSquaredError",
    metrics=["MeanSquaredError"]
)

# Fit model on training data
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_valid, y_valid))

# Evaluate neural network performance
model.evaluate(X_test,  y_test, verbose=2)
y_predict = model.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(y_predict, y_test)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1])

plt.show()

test =1