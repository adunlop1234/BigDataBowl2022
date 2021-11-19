## To Do

* Work towards a processed feature set at a point in time.

# Arthur
* Split players into offense and defence. Use a one hot encoder for offence/defence.
* Get the data that has already been dropped but required for future analysis
  * returnerId (in plays.csv), possessionTeam (plays.csv), retrieve "team" (tracking****.csv), link through returner/kicker to find the recieving team then you can allocate all of the offence/defence

# Ben
* Check that data is only included after kickoff.
* Add the kickoff location as a column
* Create a subset of the ProcessKickoffs.csv with 20 plays so that test scripts can be run on it, then push to main

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
