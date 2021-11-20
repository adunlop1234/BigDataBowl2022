from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
import sys, os
import numpy as np

# Get the size of the pixels
X_PIXELS = 128
Y_PIXELS = 64
DEPTH = 2

# Define the neural net parameters
random_state = 2022
l2_regularisation = 0.055
learning_rate = 0.0005
epochs = 500
batch_size = 16
test_size = 0.3

# Read in the data
features_filename = "features.pickle"
with open(os.path.join('..', 'processedData', features_filename), "rb") as f:
    data = pickle.load(f)

#! Work on this next
# Unpack data setting the X to be the maps and the y to be the label
# The packages seem to work fine in the end although they have the weird orange squiggle below them

y, X = [], []
for uniqueId in data.keys():
    y.append(data[uniqueId][2])
    X.append(np.stack([data[uniqueId][0], data[uniqueId][1]], axis=2))

y = np.array(y).reshape((-1,1))
X = np.array(X)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Set up the tensorflow model
model = keras.models.Sequential([
    keras.layers.Conv2D(16, kernel_size=16, activation="relu", padding="same",
                        input_shape=[X_PIXELS, Y_PIXELS, DEPTH]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, kernel_size=8, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, kernel_size=4, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation="sigmoid")
])

# Compile neural network
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Fit model on training data
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

# Evaluate neural network performance
model.evaluate(X_test,  y_test, verbose=2)
y_predict = model.predict(X_test)