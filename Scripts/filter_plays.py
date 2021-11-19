import os, sys
import numpy as np 
import pandas as pd

'''
Function to process the data for each play into cleaned formatted data.
'''

def main():

    # Get the processed data for kickoffs
    kickoffs = pd.read_csv(os.path.join("data", "ProcessedKickoffs.csv"))
    
    # Filter based on plays that have good events
    play_events_to_keep = ["kickoff_land", "kick_received", "touchback"]
    good_plays = kickoffs.loc[((kickoffs.displayName == 'football') & (kickoffs.event.isin(play_events_to_keep))), ['uniqueId']]
    good_kickoffs = kickoffs[kickoffs.uniqueId.isin(good_plays.uniqueId.values)]

    # Find where the ball landed for each play
    football = good_kickoffs[good_kickoffs.displayName == "football"]

    football_recieved = football.loc[football.event == "kick_received", ["frameId", "uniqueId"]]
    touchback = football.loc[football.event == "touchback", ["frameId", "uniqueId"]]
    football_landed = football.loc[football.event == "kickoff_land", ["frameId", "uniqueId"]]

    football_recieved = football_recieved.rename(columns={'frameId' : 'recievedFrameId'})
    touchback = touchback.rename(columns={'frameId' : 'touchbackFrameId'})
    football_landed = football_landed.rename(columns={'frameId' : 'landedFrameId'})

    # Combine to a single array that finds the x location of the ball landing
    football_all = pd.merge(football, football_recieved, on='uniqueId', how='left')
    football_all = pd.merge(football_all, touchback, on='uniqueId', how='left')
    football_all = pd.merge(football_all, football_landed, on='uniqueId', how='left')   

    # Define if the it the uniqueId lands in the endzone or not
    football_all["ballLandFrameId"] = football_all[["recievedFrameId", "touchbackFrameId", "landedFrameId"]].min(axis=1)
    endzone_lands = football_all.loc[football_all.ballLandFrameId == football_all.frameId, ["x", "uniqueId"]]
    endzone_lands["landedInEndZone"] = endzone_lands.x > 110/120

    endzone_lands.uniqueId[endzone_lands.landedInEndZone == True].to_csv('eligible_kickoff_endzone_plays.csv', index=False)


if __name__ == "__main__":
    main()