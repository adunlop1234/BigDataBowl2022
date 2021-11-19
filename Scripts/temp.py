import os
import sys
import numpy as np
import pandas as pd
import helpers 

kickoffs20 = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffsSubset20.csv'))
print(helpers.landFramePlays(kickoffs20, 128, 64))


"""
helpers.subset20Plays()
kickoffs20 = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffsSubset20.csv'))
helpers.kickoffLocationColumn(kickoffs20)
"""