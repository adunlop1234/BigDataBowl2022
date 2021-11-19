import os
import sys
import numpy as np
import pandas as pd


def subset20Plays():
    """
    Create subset of 20 plays for script testing
    """
    kickoffs = pd.read_csv(os.path.join('..', 'processedData', 'ProcessedKickoffs.csv'))
    plays20 = kickoffs["uniqueId"].unique()[:20]
    kickoffs.loc[kickoffs["uniqueId"].isin(plays20), :].to_csv(os.path.join('..', 'processedData', 'ProcessedKickoffsSubset20.csv'), index=False)

def kickoffLocationColumn(df_kickoffs):
    """
    Add column with x-distance of kickoff
    """
    kickoffLocations = df_kickoffs.loc[(df_kickoffs.event == "kickoff") & (df_kickoffs.displayName == "football"), ['x', 'uniqueId']].drop_duplicates(ignore_index=True)
    kickoffLocations = kickoffLocations.rename(columns={'x' : 'ballxKickoff'})
    kickoffs_new = pd.merge(df_kickoffs.copy(), kickoffLocations, on='uniqueId')
    
    return kickoffs_new

def binXY(kickoffs_1Play_1Frame, n_bins_x, n_bins_y):
    """
    Bins the x-y data into discrete bins
    """
    
    if (n_bins_y % 2) == 1:
        raise ValueError('y must be even')

    kickoffs_1Play_1Frame["x_bin"] = (kickoffs_1Play_1Frame["x"] * n_bins_x).astype("int32")
    kickoffs_1Play_1Frame["y_bin"] = ((kickoffs_1Play_1Frame["y"] + 0.5) * n_bins_y).astype("int32")

    print(kickoffs_1Play_1Frame.head(25))

    offence = np.zeros((n_bins_x, n_bins_y), dtype=int)
    defence = np.zeros((n_bins_x, n_bins_y), dtype=int)

    






def OLDperpareFeaturesLandFrame(kickoffs, bin_x, bin_y):
    """
    Returns a df with the features and labels, for 'land' frame only
    """
    
    # Create label
    kickoffs["Labels"] = 0
    kickoffs.loc[kickoffs["specialTeamsResult"] == "Return","Labels"] = 1

    









