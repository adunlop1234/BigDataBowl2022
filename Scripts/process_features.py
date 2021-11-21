import os, sys
import numpy as np 
import pandas as pd

'''
Function to process the data for each play into cleaned formatted data.
'''

# TODO - Filter the plays based on the desired output i.e. choose just punts, kickoff returns etc.
# TODO - Reduce the number of frames if needs be
# TODO - Adjust the start and end time of the data based on what is desired i.e. ball snap, punt kicked, punt fielded

def main():

    # Get the processed data for each year
    tracking2018 = process(os.path.join("..", "data", "tracking2018.csv"), os.path.join("..", "processedData", "eligible_kickoff_plays.csv"))
    tracking2019 = process(os.path.join("..", "data", "tracking2019.csv"), os.path.join("..", "processedData", "eligible_kickoff_plays.csv"))
    tracking2020 = process(os.path.join("..", "data", "tracking2020.csv"), os.path.join("..", "processedData", "eligible_kickoff_plays.csv"))

    # Combine data
    kickoffs = tracking2018.copy()
    kickoffs = kickoffs.append(tracking2019, ignore_index=True)
    kickoffs = kickoffs.append(tracking2020, ignore_index=True)

    # Release the memory for the big tracking files
    del tracking2018, tracking2019, tracking2020

    # Add a column which indicates what frame the ball landed
    kickoffs = add_ball_landed_column(kickoffs)

    # Add a column that is the final location of the return
    kickoffs = add_return_final_ball_location(kickoffs)

    # Write the output
    kickoffs.to_csv(os.path.join("..", 'processedData', 'ProcessedKickoffs.csv'), index=False)

    pass

def normalise_coords(df):
    '''
    Function that takes in dataframe and normalises the x and y coords between 0 and 1 (x) and -0.5 and 0.5 for (y) 
    and from left to right. Bounds of the field are 0 - 120 (x), 0 - 53.3 (y).
    '''

    # Reverse plays that go left and drop 
    df.x[df.playDirection == 'left'] = 120 - df.x[df.playDirection == 'left']
    df = df.drop(columns=['playDirection'])
    
    # Normalise x and y
    df.x = df.x / 120
    df.y = df.y / (160/3) - 0.5

    # Cap the x and y values at 0 -> 1 and -0.5 -> 0.5
    df.x[df.x > 1.0] = 1.0
    df.x[df.x < 0.0] = 0.0
    df.y[df.y > 0.5] = 0.5
    df.y[df.y < -0.5] = -0.5

    # Return dataframe 
    return df

def get_plays_information():

    # Import plays.csv 
    plays_df = pd.read_csv(os.path.join("..", "data", "plays.csv"))

    # Read in games.csv
    games_df = pd.read_csv(os.path.join("..", "data", "games.csv"))
    games_df = games_df.astype({"gameId" : str})

    # Create uniqueId on plays
    plays_df = plays_df.astype({"gameId" : str, "playId" : str})
    plays_df["uniqueId"] = plays_df.gameId + "-" + plays_df.playId

    # Join with the plays.csv on the uniqueId
    combined_df = pd.merge(plays_df, games_df, on=["gameId"])

    # Only keep relevant columns
    combined_df = combined_df[["uniqueId", "possessionTeam", "homeTeamAbbr", "visitorTeamAbbr"]]

    # Specify whether the home or away team is kicking or not
    combined_df["kickingTeam"] = combined_df.possessionTeam==combined_df.homeTeamAbbr
    combined_df.kickingTeam.replace({True: "home", False:"away"}, inplace=True)

    # Return the uniqueId, the kicking team and the returnerId to be merged later
    combined_df = combined_df[["uniqueId", "kickingTeam"]]

    return combined_df

def add_return_final_ball_location(df):

    # Get the location the ball landed in x
    ball_land_location = df.loc[(df.displayName=="football") & (df.ballLandFrameId==df.frameId), ["x", "uniqueId"]]
    ball_land_location = ball_land_location.rename(columns={"x" : "ballLandLocation"})

    # Merge the ball land location with the main dataframe
    df = pd.merge(df, ball_land_location, on="uniqueId")

    # Create new column called the final yardage line the ball is returned to
    df["returnBallFinalLocation"] = df.ballLandLocation - df.kickReturnYardage / 120
    df = df.drop(columns=["ballLandLocation"])

    return df

def insert_final_frame_Id(df):

    # Get only plays that have a retun and only the data after the catch
    returns = df.loc[df.specialTeamsResult=='Return', :]
    after_catch_data = returns.loc[returns.frameId > returns.ballLandFrameId, :]

    # Get only good plays that have the required events
    events_to_keep = ['tackle', 'out_of_bounds', 'fumble', 'touchdown']
    good_plays = after_catch_data.loc[((after_catch_data.displayName == 'football') & (after_catch_data.event.isin(events_to_keep))), ['uniqueId']]
    final_frameIds = after_catch_data[after_catch_data.uniqueId.isin(good_plays.uniqueId.values)]
    
    # Reduce data size by looking only at the football data
    final_frameIds = final_frameIds.loc[after_catch_data.displayName=='football', ["frameId", "uniqueId", "event"]]

    # Get the frameId that the specific events occur
    tackleFrameId = final_frameIds.loc[final_frameIds.event == "tackle", ["frameId", "uniqueId"]]
    outOfBoundsFrameId = final_frameIds.loc[final_frameIds.event == "out_of_bounds", ["frameId", "uniqueId"]]
    fumbleFrameId = final_frameIds.loc[final_frameIds.event == "fumble", ["frameId", "uniqueId"]]
    touchdownFrameId = final_frameIds.loc[final_frameIds.event == "touchdown", ["frameId", "uniqueId"]]

    # Rename the columns
    tackleFrameId = tackleFrameId.rename(columns={'frameId' : 'tackleFrameId'})
    outOfBoundsFrameId = outOfBoundsFrameId.rename(columns={'frameId' : 'outOfBoundsFrameId'})
    fumbleFrameId = fumbleFrameId.rename(columns={'frameId' : 'fumbleFrameId'})
    touchdownFrameId = touchdownFrameId.rename(columns={'frameId' : 'touchdownFrameId'})

    # Combine to a single array that has all of the columns
    finalFrameIdsAll = pd.merge(final_frameIds, tackleFrameId, on='uniqueId', how='left')
    finalFrameIdsAll = pd.merge(finalFrameIdsAll, outOfBoundsFrameId, on='uniqueId', how='left')
    finalFrameIdsAll = pd.merge(finalFrameIdsAll, fumbleFrameId, on='uniqueId', how='left') 
    finalFrameIdsAll = pd.merge(finalFrameIdsAll, touchdownFrameId, on='uniqueId', how='left') 

    # Find the first occurring frame
    finalFrameIdsAll["finalFrameId"] = finalFrameIdsAll[["tackleFrameId", "outOfBoundsFrameId", "fumbleFrameId", "touchdownFrameId"]].min(axis=1)
    lastFrame = finalFrameIdsAll.loc[finalFrameIdsAll.finalFrameId == finalFrameIdsAll.frameId, ["finalFrameId", "uniqueId"]]

    # Merge this data with the main dataframe
    df = pd.merge(df, lastFrame, on='uniqueId', how='left')
    return df

def add_ball_landed_column(df):
    
    # Filter based on plays that have good events
    play_events_to_keep = ["kickoff_land", "kick_received", "touchback"]
    good_plays = df.loc[((df.displayName == 'football') & (df.event.isin(play_events_to_keep))), ['uniqueId']]
    good_kickoffs = df[df.uniqueId.isin(good_plays.uniqueId.values)]

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

    # Define the ball landed frameId
    football_all["ballLandFrameId"] = football_all[["recievedFrameId", "touchbackFrameId", "landedFrameId"]].min(axis=1)
    football_all = football_all.loc[football_all.frameId==football_all.ballLandFrameId, :]

    # Merge the column for ball landed into the main dataframe
    df = pd.merge(df, football_all.loc[:, ["ballLandFrameId", "uniqueId"]], on='uniqueId')
    return df

def process(tracking_filepath, eligible_plays_filepath):

    # Process the play specified
    tracking = pd.read_csv(tracking_filepath)
    tracking = tracking.astype({"gameId" : str, "playId" : str})
    tracking["uniqueId"] = tracking.gameId + "-" + tracking.playId
    
    print("Loaded tracking data. Number of rows: " + str(len(tracking)))

    # Only keep desired columns
    tracking_columns_to_keep = ['x', 'y', 'event', 'displayName', 'position', 'frameId', 'playDirection', 'uniqueId', "team"]
    tracking = tracking[tracking_columns_to_keep]

    # Only keep the plays from the specified type list (kickoff, punts etc.)
    specific_play_ids = set((pd.read_csv(eligible_plays_filepath)).values.flatten())
    tracking_ids = set(tracking.uniqueId.unique())
    ids_to_keep = tracking_ids.intersection(specific_play_ids)
    tracking = tracking[tracking['uniqueId'].isin(ids_to_keep)]

    print("Filtered plays by kickoffs. Number of rows: " + str(len(tracking)))

    # Normalise coordinates on the play
    tracking = normalise_coords(tracking)

    print("Normalised coordinates. Number of rows: " + str(len(tracking)))

    # Drop everything that occurs before the kickoff
    kickoff_frames = tracking.loc[(tracking.displayName == 'football') & (tracking.event == 'kickoff'), ['frameId', 'uniqueId']]
    kickoff_frames = kickoff_frames.rename(columns={'frameId' : 'kickoffFrameId'})
    tracking = pd.merge(tracking.copy(), kickoff_frames, on='uniqueId')
    tracking = tracking.loc[~(tracking.frameId < tracking.kickoffFrameId), :]

    print("Dropped all frames that occur before the kickoff. Number of rows: " + str(len(tracking)))

    # Open the plays information to find the result of the plays
    plays = pd.read_csv(os.path.join("..", "data", "plays.csv"))
    plays = plays.astype({"playId" : str, "gameId" : str})
    plays["uniqueId"] = plays.gameId + "-" + plays.playId

    # Get the home/away information  
    homeAwayPossession_df = get_plays_information()

    # Combine the tracking data with the home/away team data to specify the kicking/returning team
    tracking = pd.merge(tracking, homeAwayPossession_df, on="uniqueId")
    tracking['kickingRecieving'] = tracking.team==tracking.kickingTeam
    tracking.kickingRecieving = tracking.kickingRecieving.replace({True : 'kicking', False : 'recieving'})
    tracking = tracking.drop(columns=["team", "kickingTeam"])

    # Add the relevant information from the plays and combine the dataframes
    play_columns_to_keep = ["uniqueId", "specialTeamsResult", "returnerId", "kickReturnYardage"]
    combined_data = pd.merge(tracking, plays[play_columns_to_keep], on='uniqueId')
    
    print("Merged play data. Number of rows: " + str(len(tracking)))
    return combined_data

    # Write out the processed file
    combined_data.to_csv(os.path.join('data', 'kickoffs_2020.csv'), index=False)

    print("Output the resultant file.")

    # ? CODE BLOCK 2

    # Standardise coordinates
    # * Normalise data (for better neural net performace)
    # * Set origin in x-direction as ball position in the frame previous to "ball_snap"
    # * Set origin in y-direction at the mid-pitch

    sys.exit()
    uniqueIds = combined_data.uniqueId.unique()

    for uniqueId in uniqueIds:

        current_data = combined_data[combined_data.uniqueId == uniqueId]

        for index, row in current_data.iterrows():

            # Check if next play
            if row.playId_unique != playId_track:

                # Get origin - the position of ball before snapped
                ball_start_x = combined_data.at[combined_data.index[combined_data.playId_unique == row.playId_unique].tolist()[0], "x"]
                playId_track = row.playId_unique

            # If event affecting playerss position occurs, set all values after event to zero until next play
            # E.g. on QB passing ball, all defenders rush towards ball. This will muddy the data
            if row.frameId > event_frame[row.playId_unique]:
                df_week_1_labeled.at[index, "x"] = 0
                df_week_1_labeled.at[index, "y"] = 0
                continue

            # Reverse direction of the play to make uniform left to right
            if row.playDirection == "left":        
                df_week_1_labeled.at[index, "x"] = (ball_start_x - df_week_1_labeled.at[index, "x"])
                df_week_1_labeled.at[index, "y"] = (160/3 - df_week_1_labeled.at[index, "y"]) 

            # Standardise (x,y) to be between 0.0 and 1.0
            df_week_1_labeled.at[index, "x"] = (df_week_1_labeled.at[index, "x"] - ball_start_x) / 40
            df_week_1_labeled.at[index, "y"] = df_week_1_labeled.at[index, "y"] / (160/3) - 0.5

    # Only keep relevant columns
    df_week_1_labeled = df_week_1_labeled[["x","y","position","frameId", "playId_unique", "coverage"]]

    # Ensure all data types are correct
    df_week_1_labeled = df_week_1_labeled.astype({"x" : np.float64, "y" : np.float64, "position" : str, "frameId" :str, "playId_unique" : str, "coverage" : str})

    ###? CODE BLOCK 3

    # Create new dataframe with only one row per play

    # Create column names
    columns = []
    for frame in framesIds:
        for pos in ["QB", "RB1", "RB2", "TE1", "TE2", "TE3", "WR1", "WR2", "WR3", "WR4", "LB1", "LB2", "LB3", "LB4", "LB5", "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7"]:
            for col in ["x","y"]:
                columns.append(col + "_f" + str(frame) + "_" + pos)
            if pos in ["LB1", "LB2", "LB3", "LB4", "LB5", "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7"]:
                columns.append("dist_near_off_play" + "_f" + str(frame) + "_" + pos)
                columns.append("dist_near_2_def_play" + "_f" + str(frame) + "_" + pos)
                columns.append("dist_near_3_def_play" + "_f" + str(frame) + "_" + pos)
    columns.append("coverage") 

    # Create the dataframe with indicies being the playIds_unique
    df = pd.DataFrame(columns=columns, index=set(list(df_week_1_labeled.playId_unique)))

    # Write a helper function to avoid repeating code
    def populate(row, pos_count, frameId_track):

        position = row.position
        if position in ["HB", "FB"]:
            position = "RB"

        # For simplicity, defenders are catagorised as following:
        # (it is assumed that for the rare occurrences of DE, DL and NTs, they are playing a linebacker role)
        if position in ["MLB", "OLB", "ILB", "LB", "DL", "NT", "DE"]:
            position = "LB"
        if position in ["SS", "FS", "S", "CB", "DB"]:
            position = "DB"

        if position == "QB":
            for col in ["x","y"]:
                df.at[row.playId_unique, col + "_f" + row.frameId + "_" + position] = row[col]
                return

        # Check if new frame
        if row.frameId != frameId_track[position]:
            pos_count[position] = 1
            frameId_track[position] = row.frameId
        else:
            # Same frame, so need to update position count
            pos_count[position] += 1

        for col in ["x","y"]:
                df.at[row.playId_unique, col + "_f" + row.frameId + "_" + position + str(pos_count[position])] = row[col]


    # Populate
    playId_track = 0
    for index, row in df_week_1_labeled.iterrows():

        # Check if next play
        if row.playId_unique != playId_track:
            df.at[row.playId_unique, "coverage"] = row.coverage
            playId_track = row.playId_unique
            # Reset position count
            pos_count = {"RB" : 0, "TE" : 0, "WR" : 0, "LB" : 0, "DB" : 0}
            # Reset frameId tracking
            frameId_track = {"RB" : 0, "TE" : 0, "WR" : 0, "LB" : 0, "DB" : 0}

        # Call helper function
        populate(row, pos_count, frameId_track)


    ###? CODE BLOCK 4

    # Calculate metrics used as features for neural net

    for index, row in df.iterrows():

        for frameId in framesIds:

            ## 1) 'dist_near_off_play'
            # Calculate the distance of nearest offensive player to each defender (dist_near_off_play)

            # Get offensive players' positions
            off_players = []
            for pos in ["QB", "RB1", "RB2", "TE1", "TE2", "TE3", "WR1", "WR2", "WR3", "WR4"]:

                # Check if position exists for play
                if not pd.isnull(row["x_f" + str(frameId) + "_" + pos]):
                    off_players.append(np.asarray((row["x_f" + str(frameId) + "_" + pos], row["y_f" + str(frameId) + "_" + pos])))

            # For each defensive player, find the shortest distance to nearest offensive player
            for pos in ["LB1", "LB2", "LB3", "LB4", "LB5", "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7"]:

                # Check if position exists for play
                if pd.isnull(row["x_f" + str(frameId) + "_" + pos]):
                    continue
                
                # Get position of current defender
                def_pos = np.asarray((row["x_f" + str(frameId) + "_" + pos], row["y_f" + str(frameId) + "_" + pos]))

                # Find shortest distance to offensive player
                distance = 100
                for off in off_players:
                    dist_off = np.linalg.norm(def_pos - off)
                    if dist_off < distance:
                        distance = dist_off

                df.at[index, "dist_near_off_play_f" + str(frameId) + "_" + str(pos)] = distance


            # 2) 'dist_near_2_def_play' & 'dist_near_3_def_play'

            # Store positions of all defensive players on pitch
            positions = dict()
            for pos in ["LB1", "LB2", "LB3", "LB4", "LB5", "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7"]:

                # Check if position exists for play
                if pd.isnull(row["x_f" + str(frameId) + "_" + pos]):
                    continue

                positions[pos] = np.asarray((row["x_f" + str(frameId) + "_" + pos], row["y_f" + str(frameId) + "_" + pos]))

            # Loop over each player in defence
            for player in set(positions.keys()):

                distances = []

                # Loop over teammates of current player
                for teammate in set(positions.keys()) - set(player):
                    # Add distance between player and teammate to list
                    distances.append(np.linalg.norm(positions[player] - positions[teammate]))
                distances.sort()

                df.at[index, "dist_near_2_def_play_f" + str(frameId) + "_" + player] = sum(distances[:2])/2
                df.at[index, "dist_near_3_def_play_f" + str(frameId) + "_" + player] = sum(distances[:3])/3

    # Save CSV
    df.to_csv("features.csv")

if __name__ == "__main__":
    main()