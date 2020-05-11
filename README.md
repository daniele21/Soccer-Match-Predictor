# Soccer Match Prediction Model

### Dataset:
from https://www.football-data.co.uk/

## Description:
Matematical Model based on offensive and defensive power related to each team, taking into account the home factor. The offensive power is treated as the capability of scoring goals, while the defensive one is the capability of conceding goals. The power is modeled as Poisson Distribution.

## Goals:
  - Computing the Probability and Real Odd of each event
  - Comparing estimated Odds with the bookmakers ones

## Probabilities and Odds:
Automatic generation of three excel files which provides the probabilities and odds for the following events:
  - 1, x, 2
  - 1x, x2, 12
  - Under 1.5, Under 2.5, Under 3.5
  - Over 1.5, Over 2.5, Over 3.5
  - Goal, NoGoal
  - Multigoal, Multigoal Home, Multigoal Away

## Running details:
  - Run from terminal './run_Soccer_Prediction.py'
  - Choose the championship and fill the hometeam and awayteam 
  - In the Result folder you will find the computed analysis

