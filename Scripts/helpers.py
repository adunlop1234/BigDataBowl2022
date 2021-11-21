import os
import sys
import numpy as np
import pandas as pd
import pickle


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


def lagFrames(kickoffs, n_bins_x, n_bins_y, frames, step=5, yardage=False):

    output = []

    output = {id_i : [] for id_i in kickoffs.loc[(kickoffs.specialTeamsResult=='Return')]["uniqueId"].unique()}
    
    for frame in range(frames):

        ballLandedData = kickoffs.loc[(kickoffs.frameId==kickoffs.ballLandFrameId - frame * step) & (kickoffs.specialTeamsResult=='Return'), :]
        playersData = ballLandedData.loc[(kickoffs.displayName!='football'), :]
        footballData = ballLandedData.loc[(kickoffs.displayName=='football'), :]

        data = landFramePlays(playersData, n_bins_x, n_bins_y, yardage, football=footballData)
        for id_i in data.keys():
            output[id_i].append(data[id_i])

    features_filename = "yardagePredictFeaturesFootball_Lag.pickle"
    with open(os.path.join('..', 'processedData', features_filename), "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)


def landFramePlays(df_kickoffs, n_bins_x, n_bins_y, yardage=False, football=pd.DataFrame()):
    """
    Returns a dictionary of {"uniqueId" : [kicking, recieving, label]}
    """

    listUniqueIds = df_kickoffs["uniqueId"].unique()
    data = dict()

    for id_i in listUniqueIds:
        data[id_i] = binXY(df_kickoffs.loc[(df_kickoffs["uniqueId"] == id_i), :], n_bins_x, n_bins_y, yardage, football.loc[(football["uniqueId"] == id_i), :])

    return data


def binXY(kickoffs_1Play_1Frame, n_bins_x, n_bins_y, yardage=False, football=pd.DataFrame()):
    """
    Bins the x-y data into discrete bins for one frame of one play
    """

    if (n_bins_y % 2) == 1:
        raise ValueError('y must be even')

    if not football.empty:
        df_football = football.loc[:, ["x", "y"]]
    df_kicking = kickoffs_1Play_1Frame.loc[kickoffs_1Play_1Frame["kickingRecieving"] == "kicking", ["x", "y"]]
    df_recieving = kickoffs_1Play_1Frame.loc[kickoffs_1Play_1Frame["kickingRecieving"] == "recieving", ["x", "y"]]

    tol = 0.00000001
    df_kicking["x_bin"] = ((df_kicking["x"] - tol) * n_bins_x).astype("int")
    df_kicking["y_bin"] = ((df_kicking["y"] - tol + 0.5) * n_bins_y).astype("int")
    df_recieving["x_bin"] = ((df_recieving["x"] - tol) * n_bins_x).astype("int")
    df_recieving["y_bin"] = ((df_recieving["y"] - tol + 0.5) * n_bins_y).astype("int")
    if not football.empty:
        df_football["x_bin"] = ((df_football["x"] - tol) * n_bins_x).astype("int")
        df_football["y_bin"] = ((df_football["y"] - tol + 0.5) * n_bins_y).astype("int")

    kicking = np.zeros((n_bins_x, n_bins_y), dtype=int)
    recieving = np.zeros((n_bins_x, n_bins_y), dtype=int)
    if not football.empty:
        football_out = np.zeros((n_bins_x, n_bins_y), dtype=int)

    df_kicking_np = df_kicking[["x_bin", "y_bin"]].to_numpy()
    for i in np.ndindex(df_kicking_np.shape[0]):
        kicking[df_kicking_np[i][0], df_kicking_np[i][1]] += 1   

    df_recieving_np = df_recieving[["x_bin", "y_bin"]].to_numpy()
    for i in np.ndindex(df_recieving_np.shape[0]):
        recieving[df_recieving_np[i][0], df_recieving_np[i][1]] += 1

    if not football.empty:
        df_football_np = df_football[["x_bin", "y_bin"]].to_numpy()
        for i in np.ndindex(df_football_np.shape[0]):
            football_out[df_football_np[i][0], df_football_np[i][1]] += 1

    if yardage:
        label = kickoffs_1Play_1Frame.returnBallFinalLocation.iloc[0]
    else:
        label = (int)(kickoffs_1Play_1Frame["specialTeamsResult"].iloc[0]  == "Return")

    if not football.empty:
        return (kicking, recieving, football_out, label)

    return (kicking, recieving, label)


def dataAugment(X, y):
    """
    Flip data in y-direction to double data
    """
    X_aug = np.concatenate((np.flip(X, 1), X), axis=0)
    y_aug = np.concatenate((y, y), axis=0)
    
    return (X_aug, y_aug)


def maxMinXY(df):

    df = df.loc[(df.displayName != 'football') & (df.frameId == df.ballLandFrameId), :]
    df.y.hist(bins=100)
    import matplotlib.pyplot as plt
    plt.show()

    print("Maximum x value is : " + str(df.x.max()))
    print("Minimum x value is : " + str(df.x.min()))
    print("Maximum y value is : " + str(df.y.max()))
    print("Minimum y value is : " + str(df.y.min()))


    









