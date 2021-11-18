import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import numpy as np
import sys, os

ANIMATE = True

def main():

    # Load the play dataframe
    df = pd.read_csv(os.path.join('data', 'ProcessedKickoffs.csv'))
    #df = pd.read_csv('test_play.csv')

    # Set specific play if requested
    playId = 3154
    gameId = 2020091401
    '''
    if len(sys.argv) == 2:
        playId, gameId = map(int, sys.argv[1].split('-'))

    # Display random play if not specified
    elif len(sys.argv) == 1:
        gameId = np.random.choice(df["gameId"].unique())
        playId = np.random.choice(df.loc[df["gameId"] == gameId, "playId"].unique())

    else:
        print("Usage: python plotplay.py playId-gameId")
        sys.exit(1)
    '''

    # Get all plays from the game and sort based on frameId
    if "uniqueId" in df.columns:
        plays = df[(df.uniqueId == uniqueId)].sort_values("frameId", ascending=True)
    else:
        plays = df[(df.playId == playId) & (df.gameId == gameId)].sort_values("frameId", ascending=True)
    if not len(plays):
        raise ValueError('gameId and playId are not in this year data.')

    # Normalise the play direction
    if (plays.playDirection == "left").any(): 
        plays.x = 120 - plays.x
        plays.y = 160/3 - plays.y 

    frames = plays["frameId"].unique()

    # Create title with relevant information #

    # Get home and away team
    game_df = pd.read_csv(os.path.join("data", "games.csv"))
    home = game_df.homeTeamAbbr[game_df.gameId == gameId].values[0]
    away = game_df.visitorTeamAbbr[game_df.gameId == gameId].values[0]

    # Get team in possession and opponent
    plays_df = pd.read_csv(os.path.join("data", "plays.csv"))
    offence_team = plays_df.possessionTeam[(plays_df.gameId == gameId) & (plays_df.playId == playId)].values[0]
    defence_team = home if offence_team == away else away

    # Add line of scrimage and first down marker
    line_of_scrimage = plays.loc[(plays.displayName == "football") & (plays.frameId == 1), "x"].values
    if line_of_scrimage < 110:
        first_down_marker = line_of_scrimage + 10

    # Strip the players
    players = plays.loc[plays["frameId"] == 1, "displayName"]

    # Set title
    title = offence_team + " vs. " + defence_team + " "

    fig = plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
    
    def animate(frame):

        # Increase counter to match the frame numbering
        frame += 1

        # Redraw the background each frame
        plt.clf()

        # Set up initial figure
        ax = plt.gca()
        ax.set_facecolor((0.0, 0.8, 0.0))
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 160/3)

        # Shade end-zone
        plt.axvspan(0, 10, facecolor=(0.0, 0.4, 0.0))
        plt.axvspan(110, 120, facecolor=(0.0, 0.4, 0.0))

        # Create yardage lines on the field
        for fieldline in np.linspace(10, 110, 21):
            plt.plot([fieldline, fieldline], [0, 160/3], color='w', linewidth=1.0)
        plt.plot([line_of_scrimage, line_of_scrimage], [0, 160/3], color=(0.0, 0.0, 1.0), linewidth=1.0)
        plt.plot([first_down_marker, first_down_marker], [0, 160/3], color=(1.0, 1.0, 0.0), linewidth=1.0)

        # Plot title
        plt.title(title)

        # Get data for current frame        
        frameInfo = plays[plays["frameId"] == frame]

        for player in players:

            # Get current coords
            x = frameInfo.loc[frameInfo["displayName"] == player, "x"]
            y = frameInfo.loc[frameInfo["displayName"] == player, "y"]

            # Set colour based on team/football
            team = frameInfo.loc[frameInfo["displayName"] == player, "team"].values[0]
            if team == "home":
                colour = 'b'
                markersize = 4
            elif team == "away":
                colour = 'r'
                markersize = 4
            elif team == "football":
                colour = 'k'
                markersize = 2
            else:
                print("Team ID not recognised.")
                sys.exit()

            plt.plot(x, y, color=colour, marker='o', markersize=2)

        plt.draw()
    
    if ANIMATE:
        anim = animation.FuncAnimation(fig, animate, frames=frames[-1]-1)

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10.0, metadata=dict(artist='Me'), bitrate=1800)
        anim.save("play.mp4", writer=writer)
    else:
        for frame in frames:
            animate(frame)
            plt.pause(0.001)

if __name__ == "__main__":
    main()