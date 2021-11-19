import os, sys
import pandas as pd
import helpers 
import pickle

# Read in the processed data
kickoffs = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffs.csv'))

# Get the label, 2D grid of number of recieving players and kickoff players
ballLandedData = kickoffs.loc[(kickoffs.frameId==kickoffs.ballLandFrameId) & (kickoffs.displayName!='football'), :]
output = helpers.landFramePlays(ballLandedData, 128, 64)

# Save the output
features_filename = "features.pickle"
with open(os.path.join('..', 'processedData', features_filename), "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)