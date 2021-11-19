import os
import sys
import numpy as np
import pandas as pd
import helpers 


print(pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffs.csv'))["specialTeamsResult"].unique())
"""
kickoffs20 = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffsSubset20.csv'))
helpers.kickoffLocationColumn(kickoffs20)"""