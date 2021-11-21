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


def OLDlagFrames(kickoffs, n_bins_x, n_bins_y, frames, jump, step=5):

    output = {id_i : [] for id_i in kickoffs.loc[(kickoffs.specialTeamsResult=='Return')]["uniqueId"].unique()}
    
    for frame in range(frames):

        kickoffs_frame = kickoffs.loc[(kickoffs.frameId==kickoffs.ballLandFrameId + (jump - frame) * step) & (kickoffs.specialTeamsResult=='Return'), :]
        df_players = kickoffs_frame.loc[kickoffs_frame.displayName!='football', :]
        df_football = kickoffs_frame.loc[kickoffs_frame.displayName=='football', :]

        data = framePlays(df_players, df_football, n_bins_x, n_bins_y)
        for id_i in data.keys():
            output[id_i].append(data[id_i])

    features_filename = "yardagePredictFeaturesFootball_Lag_j" + str(jump) + ".pickle"
    with open(os.path.join('..', 'processedData', features_filename), "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)


def OLDframePlays(frame_players, frame_football, n_bins_x, n_bins_y):
    """
    For one frame of input, returns a dictionary of {"uniqueId" : [kicking, recieving, football, label]}
    """

    listUniqueIds = frame_players["uniqueId"].unique()
    data = dict()

    for id_i in listUniqueIds:
        data[id_i] = binXY(frame_players.loc[(frame_players["uniqueId"] == id_i), :], frame_football.loc[(frame_football["uniqueId"] == id_i), :], n_bins_x, n_bins_y)

    return data


# TODO: Figure out why this is taking so long to run. It would take about 10 hours to complete on BC's work laptop. Probably shouldnt be that long.
def structuredData(kickoffs, n_bins_x, n_bins_y, stepSize, framesBefore, framesAfter):
    """
    Generate the data in structure: { uniqueId : { frameId : (64x32x3 , label) } }
    """

    listUniqueIds = list(kickoffs.loc[kickoffs.specialTeamsResult=='Return']["uniqueId"].unique())

    data = {id_i : 0 for id_i in listUniqueIds}

    for id_i in listUniqueIds:
        catchFrameId = kickoffs.loc[(kickoffs.uniqueId == id_i)].ballLandFrameId.iloc[0]
        kickoffFrameId = kickoffs.loc[(kickoffs.uniqueId == id_i)].kickoffFrameId.iloc[0]
        finalFrameId = kickoffs.loc[(kickoffs.uniqueId == id_i)].finalFrameId.iloc[0]
        framesKeep = [catchFrameId - delta*stepSize for delta in range(1, framesBefore+1)] + [catchFrameId] + [catchFrameId + delta*stepSize for delta in range(1, framesAfter+1)]
        framesKeep = [frame for frame in framesKeep if (frame >= kickoffFrameId and frame <= finalFrameId)]
        
        data[id_i] = {frame_i : 0 for frame_i in framesKeep}

        for frame_i in framesKeep:
            kickoffs_frame = kickoffs.loc[(kickoffs.uniqueId == id_i) & (kickoffs.frameId==frame_i), :]
            players_1frame1play = kickoffs_frame.loc[kickoffs_frame.displayName!='football', :]
            football_1frame1play = kickoffs_frame.loc[kickoffs_frame.displayName=='football', :]
            if len(players_1frame1play) > 0:
                data[id_i][frame_i] = binXY(players_1frame1play, football_1frame1play, n_bins_x, n_bins_y)

    filename = "structuredData.pickle"
    with open(os.path.join('..', 'processedData', filename), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # TODO:
    # - If frame doesnt exist for one playId will be 0 value, so need to go through and delete
    # - Keeping all frames, but need to not use garbage data before catch as will make model worse
    


def binXY(players_1frame1play, football_1frame1play, n_bins_x, n_bins_y):
    """
    Bins the x-y data into discrete bins for one frame of one play
    """

    if (n_bins_y % 2) == 1:
        raise ValueError('y must be even')

    df_football = football_1frame1play.loc[:, ["x", "y"]]
    df_kicking = players_1frame1play.loc[players_1frame1play["kickingRecieving"] == "kicking", ["x", "y"]]
    df_recieving = players_1frame1play.loc[players_1frame1play["kickingRecieving"] == "recieving", ["x", "y"]]

    tol = 0.00000001
    df_kicking["x_bin"] = ((df_kicking["x"] - tol) * n_bins_x).astype("int")
    df_kicking["y_bin"] = ((df_kicking["y"] - tol + 0.5) * n_bins_y).astype("int")
    df_recieving["x_bin"] = ((df_recieving["x"] - tol) * n_bins_x).astype("int")
    df_recieving["y_bin"] = ((df_recieving["y"] - tol + 0.5) * n_bins_y).astype("int")
    df_football["x_bin"] = ((df_football["x"] - tol) * n_bins_x).astype("int")
    df_football["y_bin"] = ((df_football["y"] - tol + 0.5) * n_bins_y).astype("int")

    kicking = np.zeros((n_bins_x, n_bins_y), dtype=int)
    recieving = np.zeros((n_bins_x, n_bins_y), dtype=int)
    football = np.zeros((n_bins_x, n_bins_y), dtype=int)

    df_kicking_np = df_kicking[["x_bin", "y_bin"]].to_numpy()
    for i in np.ndindex(df_kicking_np.shape[0]):
        kicking[df_kicking_np[i][0], df_kicking_np[i][1]] += 1   

    df_recieving_np = df_recieving[["x_bin", "y_bin"]].to_numpy()
    for i in np.ndindex(df_recieving_np.shape[0]):
        recieving[df_recieving_np[i][0], df_recieving_np[i][1]] += 1

    df_football_np = df_football[["x_bin", "y_bin"]].to_numpy()
    for i in np.ndindex(df_football_np.shape[0]):
        football[df_football_np[i][0], df_football_np[i][1]] += 1

    label = players_1frame1play.returnBallFinalLocation.iloc[0]

    return (np.stack((kicking, recieving, football), axis=2), label)


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


    









