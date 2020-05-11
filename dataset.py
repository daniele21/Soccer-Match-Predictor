#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:35:31 2019

@author: daniele
"""

#%% IMPORTS 
import pandas as pd
import numpy as np
import os 
from datetime import datetime 
from math import ceil, floor
#%% CONSTANTS
#
#serie_A = ['Data/Italy/Serie_A/I1_14-15.csv',
#           'Data/Italy/Serie_A/I1_15-16.csv',
#           'Data/Italy/Serie_A/I1_16-17.csv',
#           'Data/Italy/Serie_A/I1_17-18.csv',
#           'Data/Italy/Serie_A/I1_18-19.csv']             
#
#serieA_19 = 'Data/Italy/Serie_A/I1_18-19.csv'
#serieA_18 = 'Data/Italy/Serie_A/I1_17-18.csv'
#serieA_17 = 'Data/Italy/Serie_A/I1_16-17.csv'
#serieA_16 = 'Data/Italy/Serie_A/I1_15-16.csv'
#serieA_15 = 'Data/Italy/Serie_A/I1_14-15.csv'
#
#serie_a= ['https://www.football-data.co.uk/mmz4281/1920/I1.csv',
#             'https://www.football-data.co.uk/mmz4281/1819/I1.csv',
#             'https://www.football-data.co.uk/mmz4281/1718/I1.csv',
#             'https://www.football-data.co.uk/mmz4281/1617/I1.csv',
#             'https://www.football-data.co.uk/mmz4281/1516/I1.csv',
#             'https://www.football-data.co.uk/mmz4281/1415/I1.csv']

#%% FUNCTIONS

def computeOutcome(team, row):
        
    '''
    
        Return: Outcome(W/D/L), Points(3/1/0), Goals
    
    '''
    
    result = row.Result
    
    if(row.HomeTeam == team):
        
        if(result == '1'):
            return 'W', 3, row.HomeGoals
        elif(result == 'X'):
            return 'D', 1, row.HomeGoals
        elif(result == '2'):
            return 'L', 0, row.HomeGoals
        else:
            return 'Error', -1, -1
        
    elif(row.AwayTeam == team):
        
        if(result == '2'):
            return 'W', 3, row.AwayGoals
        elif(result == 'X'):
            return 'D', 1, row.AwayGoals
        elif(result == '1'):
            return 'L', 0, row.AwayGoals
        else:
            return 'Error', -1, -1
        
    else:
        print(team)
        print(row)
        raise Exception('Wrong Outcome')

def loadData(league_name, pathFiles):
    
    data = pd.DataFrame()
    count = len(pathFiles)
    
    for file in pathFiles:
        print('Loading...')
        print(file)
        print('')
        newData = pd.read_csv(file)
#        index = newData.index
        newData.insert(0, 'Season', count)
        newData.insert(0, 'League', league_name)
        newData.insert(2, 'Match_n', np.arange(1, len(newData)+1, 1))
        newData = newData.drop(columns=['Div'])
        data = data.append(newData, sort=False)
        count -= 1
#        print(len(data))3
    data = data.dropna(how='all')
    print(data)
    return data.reset_index()

#%%
class DatabaseManager():
    
    def __init__(self, league_name, dataFile):
        
        self.original_data = loadData(league_name, dataFile)      
        self.data = self.cleanData()
        self.findNteams()
        self.addRound()
#        self.study_data = self.genStudyData()
#        self.modelData = self.
        
#        df = pd.read_csv('df.csv', index_col=0)
    
    def cleanData(self):
        
        self.original_data.Date = pd.to_datetime(self.original_data.Date, dayfirst=True)
        
        dataset = self.original_data.iloc[:,0:11]
        dataset['Bet_1'] = self.original_data['B365H']
        dataset['Bet_X'] = self.original_data['B365D']
        dataset['Bet_2'] = self.original_data['B365A']
#        dataset.head()
        
#        dataset['Result0'] = dataset.FTR
#        dataset['Result1'] = dataset.HTR
        
        dataset.loc[dataset.FTR == 'H', ['FTR']] = '1'
        dataset.loc[dataset.FTR == 'D', ['FTR']] = 'X'
        dataset.loc[dataset.FTR == 'A', ['FTR']] = '2'
        
        dataset = dataset.rename(columns={'FTR':'Result'})
        dataset = dataset.rename(columns={'FTHG':'HomeGoals'})
        dataset = dataset.rename(columns={'FTAG':'AwayGoals'})
        
#        dataset = dataset.drop(columns=['HTHG', 'HTAG', 'HTR'])
#        dataset = dataset.drop(columns=['HTHG', 'HTAG'])
#        dataset = dataset.drop(columns=['HTHG'])
        
        dataset['payout'] = 0
        dataset.loc[dataset.Result == '1', ['payout']] = dataset['Bet_1']
        dataset.loc[dataset.Result == 'X', ['payout']] = dataset['Bet_X']
        dataset.loc[dataset.Result == '2', ['payout']] = dataset['Bet_2']
#        dataset.head()
        
        
        fav_bet = dataset[['Bet_1', 'Bet_X', 'Bet_2']].min(axis=1)
        dataset['Fav_bet'] = fav_bet
        dataset['win_fav'] = 'None'
        dataset.loc[dataset.Fav_bet == dataset.payout, ['win_fav']] = 'SI'
        dataset.loc[dataset.Fav_bet != dataset.payout, ['win_fav']] = 'NO'
#        dataset.head()
        
        
        dataset['Prob_1'] = 1/dataset.Bet_1
        dataset['Prob_X'] = 1/dataset.Bet_X
        dataset['Prob_2'] = 1/dataset.Bet_2
        dataset['Tot_prob'] = dataset.Prob_1 + dataset.Prob_2 + dataset.Prob_X
        dataset['Prob_1'] = dataset.Prob_1/dataset.Tot_prob
        dataset['Prob_X'] = dataset.Prob_X/dataset.Tot_prob
        dataset['Prob_2'] = dataset.Prob_2/dataset.Tot_prob
#        dataset.head()
#        print(dataset.dropna(how='all'))
        return dataset.dropna(how='all')
    
    def findNteams(self):
        data = self.original_data
        
        last_season = data.loc[data.Season == max(data.Season)]
        self.teams = last_season.HomeTeam.unique()
    
    def addRound(self):
        
        rounds = []
        
        for season in range(1, len(self.data.Season.unique())+1):
            
            data = self.data.loc[self.data.Season == season, :]
#            print(data)
#            print(len(data))
            for i in range(len(data)):
                index = data.iloc[i]['Match_n']
                n_round = ceil( 2*index / len(self.teams))
                rounds.append(n_round)
            
#            return data
#        print(len(rounds))
#        print(len(self.data))
        self.data.insert(3, 'Round', rounds)    
        
        
    def genStudyData(self):
       
        data = pd.DataFrame()
        count = 0
#        season = 0
        for date in self.data.Date.unique():
            
            rows = self.data.loc[self.data.Date == date]
#            print(rows)
#            return rows
#            print('Date: ', count)
            count += 1
            
            for i in range(len(rows)):
#                index = rows.iloc[i]['index']
                teamHome = rows.iloc[i].HomeTeam
                teamAway = rows.iloc[i].AwayTeam
                match = rows.iloc[i].Match_n
                season = rows.iloc[i].Season
                n_round = rows.iloc[i].Round
                
                home_res, home_res_num, home_goals = computeOutcome(teamHome, rows.iloc[i])
                away_res, away_res_num, away_goals = computeOutcome(teamAway, rows.iloc[i])
                
                df = pd.DataFrame({'Season':[season, season],
                                   'Match':[match, match],
                                   'Round':[n_round, n_round],
                                   'Date':[date, date],
                                   'Team':[teamHome, teamAway],
                                   'Goals_scored':[home_goals, away_goals],
                                   'Goals_conceded':[away_goals, home_goals],
                                   'Result':[home_res, away_res],
                                   'Points':[home_res_num, away_res_num],
                                   'Field': ['Home', 'Away']})
                data = data.append(df)
                
#            print(count)
        return data
        #trend = trend.reset_index()
        
    def getUsefulData(self):
        data = self.data.loc[:,['Date', 'Round',
                               'HomeTeam', 'AwayTeam',
                               'HomeGoals', 'AwayGoals']]
        
        return data
    
def splitData(data, season, nRound, nMatchesAhead):
    seasons = len(data.Season.unique())
    
    if(season > seasons or season < 0):
        raise Exception('Invalid season provided')
    
    df = data.loc[data.Season == season]
    test_df = df.loc[df.Round == nRound]
    
    index = test_df.index[0]
    
    df = data.loc[data.index < index]
    
    if(nMatchesAhead > index):
        train_df = df.iloc[0:index]
    else:
        train_df = df.iloc[index - nMatchesAhead - 1 : index]
    
    return train_df, test_df
#%%
class Operations():

#    def __init__(self):
               
    def findPreviousMatchesHome(df, team, date):
        
        match_home = df.dataset.loc[df.dataset.HomeTeam == team]
        prev_home_matches = match_home.loc[match_home.Date < date]
        
        return prev_home_matches.sort_values(by='Date', ascending=False)
    
    def findPreviousMatchesAway(df, team, date):
        
        match_away = df.dataset.loc[df.dataset.AwayTeam == team]
        prev_away_matches = match_away.loc[match_away.Date < date]
        
        return prev_away_matches.sort_values(by='Date', ascending=False)
    
    def findPreviousMatches(df, team, date):
        
        prev_matches = pd.DataFrame()
        prev_home_matches = df.findPreviousMatchesHome(team, date)
        prev_away_matches = df.findPreviousMatchesAway(team, date)
        
        prev_matches = prev_matches.append(prev_home_matches)
        prev_matches = prev_matches.append(prev_away_matches)
        
        return prev_matches.sort_values(by='Date', ascending=False)
    
    def setTrend(df, team, date, prev_matches, label):
        
        day_matches = df.loc[df.Date == date]
        row = day_matches.loc[day_matches.Team == team]
        index = row.index
        
        if(len(prev_matches) >= 5):
            n_prev_matches = 5
        else:    
            n_prev_matches = len(prev_matches)
            
        for i in range(n_prev_matches):    
            column = label + '-' + str(i+1) 
            df.loc[index, column] = computeOutcome(team, prev_matches.iloc[i])

#%%   
        
        