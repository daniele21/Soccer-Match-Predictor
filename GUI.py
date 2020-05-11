#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:44:52 2019

@author: daniele
"""

#%% IMPORTS 
import tkinter as tk
import pandas as pd

import data as T
from Prediction_Model import SoccerPrediction, selection
from Prediction_Model import getAll1X2
from Prediction_Model import getAllMultigoal
from Prediction_Model import getAllOvUnGoalNoGoal

def predictRound(leagueName, csvName, nRound, teams, opponents):
    
    model = SoccerPrediction(leagueName, csvName)    
    model.checkStatsLeague(model.db.data)
    test = pd.DataFrame({'Round': nRound,
                         'HomeTeam': teams, 'AwayTeam':opponents})
      
    unoXdue = getAll1X2(test, model)
    print('> Computing 1 X 2')
    multigoal = getAllMultigoal(test, model, save=True)
    print('> Computing MULTIGOALS')
    unov = getAllOvUnGoalNoGoal(test, model, save=True)
    print('> Computing OVER UNDER')
    
    sel = selection(model, nRound, unoXdue, unov, 0.5, 0.75, 0.75, 0.75)
    
    print('> Finish')
#%%
class GridStructure():
    
    def __init__(self, row, column, rowspan, columnspan, padx=1, pady=1):
        self.row = row
        self.column = column
        self.rowspan = rowspan
        self.columnspan = columnspan
        self.padx = padx
        self.pady = pady
    
    
def _placeOnCenter(root):

#    h = int(root.winfo_reqheight())
#    w = int(root.winfo_reqwidth())

    h = int(root.winfo_reqheight())*2
    w = int(root.winfo_reqwidth())*2
    

    
#    print(str(w) + ' - ' + str(h))
    
    # Gets both half the screen width/height and window width/height
    positionX = int(root.winfo_screenwidth()/2 - w)
    positionY = int(root.winfo_screenheight()/2 - h)

    # Positions the window in the center of the page.
    root.geometry("+{}+{}".format(positionX, positionY))
    
def _insertOptionMenu(root, var, list_elem, grid):
    
    menu = tk.OptionMenu(root, var, *list_elem)
    menu.config(font=('courier new', 10, 'bold'))
    menu.grid(row = grid.row,
                column = grid.column,
                rowspan = grid.rowspan,
                columnspan = grid.columnspan,
                padx=grid.padx,
                pady=grid.pady)
    
    return menu

def _insertMatch(root, teams_list, grid, i=0, callback=None):
    
    teamA_var = tk.StringVar(name='Home_' + i)
    teamB_var = tk.StringVar(name='Away_' + i)
    teamA_var.set('Home Team')
    teamB_var.set('Away Team')
    
    row = grid.row
    gridA = GridStructure(row, 1,1,1, grid.padx, grid.pady)
    gridvs = GridStructure(row, 2,1,1)
    gridB = GridStructure(row, 3,1,1, grid.padx, grid.pady)
    
    vsLabel = tk.Label(root, text='vs')
    
    menuA = _insertOptionMenu(root, teamA_var, teams_list, gridA)
    menuB = _insertOptionMenu(root, teamB_var, teams_list, gridB)
    
    vsLabel.grid(row = gridvs.row, 
                 column = gridvs.column)
    
    if(callback is not None):
        teamA_var.trace('w', callback)
        teamB_var.trace('w', callback)

    return menuA, menuB, teamA_var, teamB_var

def _destroyMenus(menus):
    
    for i in range(len(menus)):
        menus[i].destroy()
        

class matchChooser():
    
    def __init__(self):
        
        self.root = tk.Tk()
        _placeOnCenter(self.root)
        
        self.league_list = T.TEAMS_LEAGUE
        
        self.teams_list = []
        
        self.mainWindow()
        
        self.root.mainloop()
    
    def __leagueOnChange(self, *args):
        league_name = self.league_var.get()
        print('> Changing League: {}'.format(league_name))
        
        self.teams_list = self.league_list[league_name]
        self.teams_list.sort()
        
        if(len(self.teamsHome_menus) != 0 and len(self.teamsAway_menus) != 0):
#            print(len(self.teamsA_menus))
            _destroyMenus(self.teamsHome_menus)
            _destroyMenus(self.teamsAway_menus)
        
        for i in range(len(self.teams_list)//2):
            row = i+2
            grid = GridStructure(row,1,1,1, 25,7)
            teamA_menu, teamB_menu, varA, varB = _insertMatch(self.root, self.teams_list, grid, str(i))
            
            self.teamsHome_menus.update({i:teamA_menu})
            self.teamsAway_menus.update({i:teamB_menu})
            
            self.teamsHome.update({i:varA})
            self.teamsAway.update({i:varB})
            
        self.calculate_button.configure(state=tk.ACTIVE)
    
    def getTeams(self):
        
        assert len(self.teamsHome) == len(self.teamsAway), 'Wrong Teams'
        
        home, away = [], []
        
        for i in range(len(self.teamsHome)):
            homeTeam = self.teamsHome[i].get()
            awayTeam = self.teamsAway[i].get()
            
            if(homeTeam != 'Home Team'):
                home.append(homeTeam)
            
            if(awayTeam != 'Away Team'):
                away.append(awayTeam)
            
        return home, away
    
    def mainWindow(self):
        
        self.teamsHome_menus = {}
        self.teamsAway_menus = {}
        self.teamsHome = {}
        self.teamsAway = {}
        
        self.league_var = tk.StringVar()
        self.league_var.set('League Name')
        self.league_var.trace('w', self.__leagueOnChange)
        
        league_grid = GridStructure(1,1,1,3, 20, 20)   
        
        league_menu = _insertOptionMenu(self.root, self.league_var, T.LEAGUE_NAMES, league_grid)
        
        
        
        calculate_grid = GridStructure(league_grid.row+15, 3,1,1)
        self.calculate_button = tk.Button(self.root, text='Calcola', command=self.calculateAction)
        self.calculate_button.grid(row = calculate_grid.row, 
                              column = 2, 
                              padx = 10,
                              pady = 5)
        self.calculate_button.configure(state=tk.DISABLED)
        
        tk.Label(self.root, text='Round N.').grid(row=calculate_grid.row-1, column = 1, pady=(20,10))
        
        self.round_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.round_var,
                 justify='center',
                 width = 7).grid(row = calculate_grid.row, column = 1)
        
    def calculateAction(self, *args):
        
        homeTeams, awayTeams = self.getTeams()
        print(homeTeams)
        print(awayTeams)
        print(self.round_var.get())
        
        league = self.league_var.get()
        n_round = int(self.round_var.get())
        
        predictRound(league, T.LEAGUE_CSV[league], n_round, homeTeams, awayTeams)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        