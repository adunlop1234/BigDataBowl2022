# BigDataBowl2022
NFL Big Data Bowl 2022

# Ideas

### Punt return / Kickoff return:
We could have a collection of metrics all in two animations (one for punts and one for kickoffs):
* Probability return the punt / kickoff:
  - Up until the point where know for certain, calculate the probability a player will return the ball
* Expected yards with time
  - Calculate expected yards as a function of time
  - Calculate a measure of confidence (which will obviously start large and finish as 0). This could either be a variance, or a probability.


## Ranking players
This is relatively easy and was recieved well last year.
* At point catch the ball, ratio of gained yards to expected yards (if negative expected yards, won't work, but get the idea)
* Decision making (when returning a kickoff was better decision than kneeling in end zone)

More basic rankings:
* Best players at blocking (return team)
* Best players at evading blocking (kicking team)
* Best players at evading tackles (e.g. number of players within 0.5m of punt returner who didn't make tackle).
* Best players at making tackles (or not missing tackles)
* Best players at forcing fumbles (might not be enough for statistical significance).
