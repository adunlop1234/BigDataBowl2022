import os, sys
import pandas as pd
import helpers 
import pickle
import matplotlib.pyplot as plt

# Read in the processed data
kickoffs = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffs.csv'))
helpers.lagFrames(kickoffs, 64, 32, 5, step=4, yardage=False)


sys.exit()
'''
# Get the label, 2D grid of number of recieving players and kickoff players
ballLandedData = kickoffs.loc[(kickoffs.frameId==kickoffs.ballLandFrameId) & (kickoffs.displayName!='football'), :]

# Write the predict return data for 64, 32 
output = helpers.landFramePlays(ballLandedData, 64, 32, yardage=False)
features_filename = "returnPredictFeatures.pickle"
with open(os.path.join('..', 'processedData', features_filename), "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
'''
# Get the label, 2D grid of number of recieving players and kickoff players
ballLandedData = kickoffs.loc[(kickoffs.frameId==kickoffs.ballLandFrameId) & (kickoffs.specialTeamsResult=='Return'), :]
playersData = ballLandedData.loc[(kickoffs.displayName!='football'), :]
footballData = ballLandedData.loc[(kickoffs.displayName=='football'), :]

# Write the predict return data for 64, 32 
output = helpers.landFramePlays(playersData, 64, 32, yardage=True, football=footballData)
features_filename = "yardagePredictFeaturesFootball.pickle"
with open(os.path.join('..', 'processedData', features_filename), "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)