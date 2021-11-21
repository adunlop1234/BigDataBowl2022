import os
import sys
import numpy as np
import pandas as pd
import helpers 
import pickle
from matplotlib import pyplot as plt
import process_features
import time

kickoffs = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffs.csv'))
helpers.lagFrames(kickoffs, 64, 32, 5, step=4, yardage=False)


sys.exit()


fig, axs = plt.subplots(5, 3)
for i in range(3):
    for j in range(5):
        axs[j,i].imshow(X[0][:,:,(i+1)*(j+1)-1])
plt.show()


uniqueId = '2018123000-36'

with open(os.path.join('..', 'processedData', 'yardagePredictFeaturesFootball.pickle'), 'rb') as f:
    features = pickle.load(f)

plt.imshow(features[uniqueId][0])
plt.show()
plt.imshow(features[uniqueId][1])
plt.show()
plt.imshow(features[uniqueId][2])
plt.show()
sys.exit()

helpers.subset20Plays()
kickoffs20 = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffsSubset20.csv'))
helpers.kickoffLocationColumn(kickoffs20)


kickoffs = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffs.csv'))
start = time.time()
output = helpers.landFramePlays(kickoffs.loc[(kickoffs.frameId==kickoffs.ballLandFrameId) & (kickoffs.displayName!='football'), :], 64, 32)
end = time.time()
print(end - start)

features_filename = "features.pickle"

with open(features_filename, "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
sys.exit()

uniqueId = '2018123000-36'

with open(features_filename, 'rb') as f:
    features = pickle.load(f)

plt.imshow(features[uniqueId][0])
plt.show()
plt.imshow(features[uniqueId][1])
plt.show()


"""
helpers.subset20Plays()
kickoffs20 = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffsSubset20.csv'))
helpers.kickoffLocationColumn(kickoffs20)
"""