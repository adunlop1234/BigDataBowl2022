from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
import sys, os
import numpy as np

# Get the size of the pixels
X_PIXELS = 64
Y_PIXELS = 32
DEPTH = 2

# Define the neural net parameters
random_state = 2022
l2_regularisation = 0.055
learning_rate = 0.0005
epochs = 10
batch_size = 32
test_size = 0.1
valid_size = 0.2

# Read in the data
features_filename = "featuresRegression.pickle"
with open(os.path.join('..', 'processedData', features_filename), "rb") as f:
    data = pickle.load(f)

y, X = [], []
for uniqueId in data.keys():
    y.append(data[uniqueId][2])
    X.append(np.stack([data[uniqueId][0], data[uniqueId][1]], axis=2))

y = np.array(y)
X = np.array(X)


# Split the data
X_trainValid, X_test, y_trainValid, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainValid, y_trainValid, test_size=valid_size, random_state=random_state, stratify=y_trainValid)

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
    metrics=["accuracy"]
)

# Fit model on training data
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

# Evaluate neural network performance
model.evaluate(X_test,  y_test, verbose=2)
y_predict = model.predict(X_test)