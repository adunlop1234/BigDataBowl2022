## To Do


# Ben
Preprocessing

# Arthur
Get the return yardage featuyres into processed kickoffs
Get the label to be return yardage + yardage location of the recieiving point
Restrict the dataset used by the E(yards) model to only feature ones which were returned

Create a function to augment the training data by flipping in the y direction
Create another model that has an output of expected yard line finish

* Work towards a processed feature set at a point in time.

# Model Improvements
* Tune hyperparameters
* Cross validation

# Previous Ideas

# Create a pre-processor that will perform the following on the data:
    * Normalise the x and y coordinates to go between 0 and 1 for x and -1 and 1 for y
    * Extract data only after the snap
    * Extract data only up until and including the confirmation of return or no return
    * Add a parameter which is time remaining until the return/no return and normalise between 0 - 1 (snap - result)

# Data Augmentation
    * Data is augmented by creating another set of plays all flipped in the y direction, once the test-train-split has been complete

# Features to be used in the classifier/clustering

    * Keep:

        * Recieving Team
            * Nominal Reciever
            * Wide Blockers

        * Punting Team
            * Chasers

        * Ball

    * Blocker Features
        * x
        * y

    * Runner Features
        * Distance from nearest defender

# Models to investigate to predict outcome
    * Time Dependent Neural Nets
        * Simple RNN (No long term memory)
        * LSTM
        * GRU
    * Trees with time features:
        * Decision Tree
        * Random Forest
        * XGBoost
