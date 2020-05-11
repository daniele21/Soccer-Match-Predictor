#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:32:56 2019

@author: daniele
"""

#%% IMPORTS

from dataset import DatabaseManager, loadData, splitData
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import poisson, skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
import time

result_path = './Results/'
#%%

def rho_correction(x, y, lambda_x, mu_y, rho):
    if x==0 and y==0:
        return 1- (lambda_x * mu_y * rho)
    elif x==0 and y==1:
        return 1 + (lambda_x * rho)
    elif x==1 and y==0:
        return 1 + (mu_y * rho)
    elif x==1 and y==1:
        return 1 - rho
    else:
        return 1.0
    
def solve_parameters(dataset, debug = False, init_vals=None, options={'disp': True, 'maxiter':100},
                     constraints = [{'type':'eq', 'fun': lambda x: sum(x[:20])-20}] , **kwargs):
    
    teams = np.sort(dataset['HomeTeam'].unique())
    # check for no weirdness in dataset
    away_teams = np.sort(dataset['AwayTeam'].unique())
    if not np.array_equal(teams, away_teams):
        raise ValueError("Something's not right")
    n_teams = len(teams)
    if init_vals is None:
        # random initialisation of model parameters
        init_vals = np.concatenate((np.random.uniform(0,1,(n_teams)), # attack strength
                                      np.random.uniform(0,-1,(n_teams)), # defence strength
                                      np.array([0, 1.0]) # rho (score correction), gamma (home advantage)
                                     ))
    def dc_log_like(x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma):
        lambda_x, mu_y = np.exp(alpha_x + beta_y + gamma), np.exp(alpha_y + beta_x) 
        return (np.log(rho_correction(x, y, lambda_x, mu_y, rho)) + 
                np.log(poisson.pmf(x, lambda_x)) + np.log(poisson.pmf(y, mu_y)))
    
#    @jit(float64(float64[:], int64), nopython=True, parallel=True)
    def estimate_paramters(params):
        score_coefs = dict(zip(teams, params[:n_teams]))
        defend_coefs = dict(zip(teams, params[n_teams:(2*n_teams)]))
        rho, gamma = params[-2:]
        log_like = [dc_log_like(row.HomeGoals, row.AwayGoals, score_coefs[row.HomeTeam], defend_coefs[row.HomeTeam],
                     score_coefs[row.AwayTeam], defend_coefs[row.AwayTeam], rho, gamma) for row in dataset.itertuples()]
        return -sum(log_like)
    
#    @jit(float64[:](float64[:], int64), nopython=True, parallel=True)
#    def fast_jac(x, N):
#        h = 1e-9
#        jac = np.zeros_like(x)
#        f_0 = estimate_paramters(params)
#        for i in range(N):
#            x_d = np.copy(x)
#            x_d[i] += h
#            f_d = estimate_paramters(x_d, N)
#            jac[i] = (f_d - f_0) / h
#        return jac
    
    
    opt_output = minimize(estimate_paramters, init_vals, method='SLSQP', options=options, constraints = constraints, **kwargs)
    if debug:
        # sort of hacky way to investigate the output of the optimisation process
        return opt_output
    else:
        return dict(zip(["attack_"+team for team in teams] + 
                        ["defence_"+team for team in teams] +
                        ['rho', 'home_adv'],
                        opt_output.x))
        
def calc_means(param_dict, homeTeam, awayTeam):
    return [np.exp(param_dict['attack_'+homeTeam] + param_dict['defence_'+awayTeam] + param_dict['home_adv']),
            np.exp(param_dict['defence_'+homeTeam] + param_dict['attack_'+awayTeam])]

def dixon_coles_simulate_match(params_dict, homeTeam, awayTeam, max_goals=8):
    team_avgs = calc_means(params_dict, homeTeam, awayTeam)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([[rho_correction(home_goals, away_goals, team_avgs[0],
                                                   team_avgs[1], params_dict['rho']) for away_goals in range(2)]
                                   for home_goals in range(2)])
    output_matrix[:2,:2] = output_matrix[:2,:2] * correction_matrix
    return output_matrix

class SoccerPrediction():
    
    def __init__(self, league_name, pathFile, dixon=False):
        
        self.league_name = league_name
        self.db = DatabaseManager(league_name, pathFile)
        self.data = self.db.getUsefulData()
        self.model = self.poissonModel()
        self.teams = self.db.teams
        self.dixon = dixon
        
        if(dixon):
            self.dixon_params = self.dixon()
        
    def dixon(self):
        start = time.time()
        reduced_set = self.db.data.iloc[0:400]
#            return reduced_set
        params = solve_parameters(reduced_set)
        end = time.time()
        spent_time_min = (end - start)//60
        spent_time_sec = end - start - spent_time_min*60
        print('Time spent: {} min {} sec'.format(spent_time_min, spent_time_sec)) 

        return params
    
    def updateData(self, data):
        self.data = data
    
    def poissonModel(self):
        data = self._genData(self.data)
        
        data_home = data.rename(columns = {'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}) 
        data_home = data_home.assign(home = 1)
        
        data_away = data.rename(columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})
        data_away = data_away.assign(home = 0)
        
#        return data_home, data_away
#        goal_model_data = pd.concat([data_home, data_away])
        
        goal_model_data = pd.concat([data[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
                                    columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
    
                                   data[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
                                    columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

        poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                                family=sm.families.Poisson()).fit()
        
#        print(poisson_model.summary())  

        return poisson_model
    
    def _genData(self, data):
        
        if(data is None):
            data = self.data
        
        data = data.loc[:,['HomeTeam', 'AwayTeam',
                           'HomeGoals', 'AwayGoals']]
        
        return data
        
    def checkStatsLeague(self, data=None, plot=True):
        
        data = self._genData(data)
        
#        print(data.mean())
        
        poisson_pred = np.column_stack([[poisson.pmf(x, data.mean()[j]) for x in range(8)] for j in range(2)])
        
        # plot histogram of actual goals
        [values, bins, _] = plt.hist(data[['HomeGoals', 'AwayGoals']].values, range(9), 
                 alpha=0.7, label=['Home', 'Away'],normed=True, color=["#FFA07A", "#20B2AA"])
        
        # add lines for the Poisson distributions
        pois1, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0],
                          linestyle='-', marker='o',label="Home", color = '#CD5C5C')
                          
        pois2, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1],
                          linestyle='-', marker='o',label="Away", color = '#006400')
        
        leg=plt.legend(loc='upper right', fontsize=13, ncol=2)
        leg.set_title("Poisson           Actual        ", prop = {'size':'14', 'weight':'bold'})
        
        plt.xticks([i-0.5 for i in range(1,9)],[i for i in range(9)])
        plt.xlabel("Goals per Match",size=13)
        plt.ylabel("Proportion of Matches",size=13)
        plt.title("Number of Goals per Match",size=14,fontweight='bold')
        plt.ylim([-0.004, 0.4])
        plt.tight_layout()
        plt.show()
        
        home_error = np.mean(abs(poisson_pred[:,0] - values[0]))
        away_error = np.mean(abs(poisson_pred[:,1] - values[1]))
        
        return [home_error, away_error]
        
    def checkDiffInGoals(self, data=None):
        data = self._genData(data)
        
        skellam_pred = [skellam.pmf(i,  data.mean()[0],  data.mean()[1]) for i in range(-6,8)]
        
        plt.hist(data[['HomeGoals']].values - data[['AwayGoals']].values, range(-6,8), 
                 alpha=0.7, label='Actual',normed=True)
        plt.plot([i+0.5 for i in range(-6,8)], skellam_pred,
                          linestyle='-', marker='o',label="Skellam", color = '#CD5C5C')
        plt.legend(loc='upper right', fontsize=13)
        plt.xticks([i+0.5 for i in range(-6,8)],[i for i in range(-6,8)])
        plt.xlabel("Home Goals - Away Goals",size=13)
        plt.ylabel("Proportion of Matches",size=13)
        plt.title("Difference in Goals Scored (Home Team vs Away Team)",size=14,fontweight='bold')
        plt.ylim([-0.004, 0.26])
        plt.tight_layout()
        plt.show()

    def checkStatsMatch(self, team1, team2, data):
        
        fig,(ax1,ax2) = plt.subplots(2, 1, figsize=(7,5))

        team1_home = data[data['HomeTeam']==team1][['HomeGoals']].apply(pd.value_counts,normalize=True)
        team1_home_pois = [poisson.pmf(i,np.sum(np.multiply(team1_home.values.T,team1_home.index.T),axis=1)[0]) for i in range(8)]
        
        team2_home = data[data['HomeTeam']==team2][['HomeGoals']].apply(pd.value_counts,normalize=True)
        team2_home_pois = [poisson.pmf(i,np.sum(np.multiply(team2_home.values.T,team2_home.index.T),axis=1)[0]) for i in range(8)]
        
        team1_away = data[data['AwayTeam']==team1][['AwayGoals']].apply(pd.value_counts,normalize=True)
        team1_away_pois = [poisson.pmf(i,np.sum(np.multiply(team1_away.values.T,team1_away.index.T),axis=1)[0]) for i in range(8)]
        team2_away = data[data['AwayTeam']==team2][['AwayGoals']].apply(pd.value_counts,normalize=True)
        team2_away_pois = [poisson.pmf(i,np.sum(np.multiply(team2_away.values.T,team2_away.index.T),axis=1)[0]) for i in range(8)]
        
        ax1.bar(team1_home.index-0.4, team1_home.values.reshape(team1_home.shape[0]), width=0.4, color="#034694", label=team1)
        ax1.bar(team2_home.index,team2_home.values.reshape(team2_home.shape[0]),width=0.4,color="#EB172B",label=team2)
        pois1, = ax1.plot([i for i in range(8)], team1_home_pois,
                          linestyle='-', marker='o',label=team1, color = "#0a7bff")
        pois1, = ax1.plot([i for i in range(8)], team2_home_pois,
                          linestyle='-', marker='o',label=team2, color = "#ff7c89")
        leg=ax1.legend(loc='upper right', fontsize=12, ncol=2)
        leg.set_title("Poisson                 Actual                ", prop = {'size':'14', 'weight':'bold'})
        ax1.set_xlim([-0.5,7.5])
        ax1.set_ylim([-0.01,0.65])
        ax1.set_xticklabels([])
        # mimicing the facet plots in ggplot2 with a bit of a hack
#        ax1.text(7.65, 0, '                Home                ', rotation=-90,
#                bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})
#        ax2.text(7.65, 0, '                Away                ', rotation=-90,
#                bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})
        
        ax2.bar(team1_away.index-0.4,team1_away.values.reshape(team1_away.shape[0]),width=0.4,color="#034694",label=team1)
        ax2.bar(team2_away.index,team2_away.values.reshape(team2_away.shape[0]),width=0.4,color="#EB172B",label=team2)
        
        pois1, = ax2.plot([i for i in range(8)], team1_away_pois,
                          linestyle='-', marker='o',label=team1, color = "#0a7bff")
        pois1, = ax2.plot([i for i in range(8)], team2_away_pois,
                          linestyle='-', marker='o',label=team2, color = "#ff7c89")
        
        ax2.set_xlim([-0.5,7.5])
        ax2.set_ylim([-0.01,0.65])
        ax1.set_title("Number of Goals per Match    {} vs {}".format(team1,team2),size=14,fontweight='bold')
        ax2.set_title("Number of Goals per Match    {} vs {}".format(team2,team1),size=14,fontweight='bold')
        ax2.set_xlabel("Goals per Match",size=13)
        #ax2.text(-1.15, 0.9, 'Proportion of Matches', rotation=90, size=13)
        plt.tight_layout()
        plt.show()
        
    def drawProbLeague(self, data=None):
        data = self._genData(data)
        prob = self._probGoalsDiff(0, data)
        
        return 'Draw: {:.1f} %'.format(prob)
    
    def homeWinProbLeague(self, goal_di_scarto, data=None):
        data = self._genData(data)
        prob = self._probGoalsDiff(goal_di_scarto, data)
        
        return 'Home team wins with {} goals more, with prob: {:.1f} %'.format(goal_di_scarto, prob)

    def _probGoalsDiff(self, diff, data):
        goals_diff = diff
        return skellam.pmf(goals_diff,  data.mean()[0],  data.mean()[1])
        
    def predAvgGoalTeam(self, team, opponent, home_factor):
        
        match = {'team':team,
                 'opponent':opponent,
                 'home':home_factor}
        
        match_df = pd.DataFrame(match, index=[1])
        
        n_goals = self.model.predict(match_df)
        
        print('\nResult\n')
        print(team.upper() + ' probably will score {:.2f} goals'.format(n_goals.values[0])) 
        
        return n_goals.values[0]
        
    def simulate_match(self, homeTeam, awayTeam, max_goals=8):
        
        self.max_goals = 8
        
        if(self.dixon == True):
        
            match_outcome = dixon_coles_simulate_match(self.dixon_params, homeTeam, awayTeam)
            
            result = {'Match':[str(homeTeam + ' vs ' + awayTeam)],
                                  'Outcome': [match_outcome]}
            match = pd.DataFrame(result)
            
            return match 
        
        else:
        
            home_data = {'team':homeTeam,
                         'opponent':awayTeam,
                         'home':1}
            
            away_data = {'team':awayTeam,
                         'opponent':homeTeam,
                         'home':0}
            
            home_df = pd.DataFrame(home_data, index=[1])
            away_df = pd.DataFrame(away_data, index=[1])
            
            
            home_goals_avg = self.model.predict(home_df).values[0]
            away_goals_avg = self.model.predict(away_df).values[0]
            
            team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
            
            match_outcome = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
            
            result = {'Match':[str(homeTeam + ' vs ' + awayTeam)],
                                  'Outcome': [match_outcome]}
            match = pd.DataFrame(result)
            
            return match 
    
    def predictRound(self, test, nRound, gap=20, save=True):
        
#        season = max(self.db.data.Season)
#        nMatchAhead = len(self.db.data)
#        
#        train, test = splitData(self.db.data, season, nRound, nMatchAhead)
#        
#        data = test.loc[test.Season == season]
#        data = data.loc[data.Round == nRound]
#        self.round = nRound
        for i in range(len(test)):
            
            self.my_results = getAll1X2(test, self)
            
#            self.book = getAllBookmakersOdds(test)
            
        mix = pd.DataFrame()
        nanRow = pd.DataFrame({'Match': [np.nan],
                                '1':[np.nan], 'X':[np.nan], '2':[np.nan],
                                '1X':[np.nan], 'X2':[np.nan], '12':[np.nan]})
        
        nanRow = nanRow.rename(index={0:'_______'})
        
        i=0
        while(i <= len(self.book)):
            
#            lineRes = my_results.iloc[i:i+2]
#            lineBook = book.iloc[i:i+2]
#            
#            mix = mix.append(lineRes, sort=False)
#            mix = mix.append(lineBook, sort=False)
#            mix = mix.append(nanRow, sort=False)
#            i += 2 
            
            lineRes = self.my_results.iloc[i:i+1]
            lineBook = self.book.iloc[i:i+1]
            
            mix = mix.append(lineRes, sort=False)
            mix = mix.append(lineBook, sort=False)
            
            lineRes = self.my_results.iloc[i+1:i+2]
            lineBook = self.book.iloc[i+1:i+2]
            
            mix = mix.append(lineRes, sort=False)
            mix = mix.append(lineBook, sort=False)
            
            mix = mix.append(nanRow, sort=False)
            i += 2 
            
#        mix = mix.round(1)
        
        mix.reset_index(drop=True)
        
        if(save):    
            filename = self.league_name + '_Round_' + str(nRound) + '.xlsx' 
            mix.to_excel(filename, index=False)
        
        advs = self.checkAdvantage(gap=gap, save=save)
        
        return mix, advs
        
    def checkAdvantage(self, gap, save=True):
        
        advantages = pd.DataFrame()
        
        i = 0
#        cell_values = []
        
        while(i <= len(self.my_results)-1):
            match = self.my_results.iloc[i].Match
            
            for col in self.my_results.columns:
                if(col != 'Match' and col != 'Info'):
                    my_prob = self.my_results.iloc[i][col]
                    my_odd = self.my_results.iloc[i+1][col]
                    
                    book_prob = self.book.iloc[i][col] 
                    book_odd = self.book.iloc[i+1][col] 
                    
                    if(float(my_prob)*100 - float(book_prob)*100 > int(gap)):
                        adv = pd.DataFrame({'Match':match,
                                            col:[my_prob, book_prob, my_odd, book_odd]})
                        adv = adv.rename(index={0:'My_Prob', 1:'Book_prob',
                                          2:'My_odd', 3:'Book_odd'})
                        advantages = advantages.append(adv, sort=False)
#                        cell_values = cell_values.append()
            
            i += 2
        
        advantages.reset_index(drop=True)
        
        if(save):
            filename = self.league_name + '_Round_' + str(self.round) + '_1X2' + '.xlsx'
            advantages.to_excel(filename)
            
        return advantages
    
#%%    
def tuneModel(data):
    
    match = []
    home_error = []
    away_error = []
    
    for i in range(380, len(data), 20):
        
        train, _ = ds.splitData(data, season, nRound, i)
        error = model.checkStatsLeague(train, plot=False)
        
        match.append(i)
        home_error.append(error[0])
        away_error.append(error[1])
        
    return [match, home_error, away_error]   
  
def divideDataset(df, gap=2):    

    mix = pd.DataFrame()
        
    col = df.columns
    nanArray = np.empty((len(col),1))
    nanArray[:] = np.nan
    
    nanRow = pd.DataFrame(nanArray.transpose(), columns=col)
    
    i=0
    
    while(i <= len(df)):
#        print(len(df))
        
        for j in range(i, i+gap):
            
            lineOne = df.iloc[j : j+1]
            mix = mix.append(lineOne, sort=False)
            
        mix = mix.append(nanRow, sort=False)
#        print(mix)
#        break
        i += gap
#        print(i)
        
    return mix
    
def divideDatasets(df1, df2):
    mix = pd.DataFrame()
    
    col = df1.columns
    nanArray = np.empty((len(col),1))
    nanArray[:] = np.nan
    
    nanRow = pd.DataFrame(nanArray, columns=col)
    
    i=0
    
    while(i <= len(df1)):
        
        lineRes = df1.iloc[i:i+1]
        lineBook = df2.iloc[i:i+1]
        
        mix = mix.append(lineRes, sort=False)
        mix = mix.append(lineBook, sort=False)
        
        lineRes = df1.iloc[i+1:i+2]
        lineBook = df2.iloc[i+1:i+2]
        
        mix = mix.append(lineRes, sort=False)
        mix = mix.append(lineBook, sort=False)
        
        mix = mix.append(nanRow, sort=False)
        i += 2 
        
    return mix
      
def get1X2(match_outcome):

    outcome = match_outcome.Outcome.values[0]
#    print(outcome)
    # HOME WIN
    _1 = np.sum(np.tril(outcome, -1))
    
    # DRAW
    _X = np.sum(np.diag(outcome))
    
    # AWAY WIN
    _2 = np.sum(np.triu(outcome, 1))       
    
#    probs = {'Match': match_outcome.Match[0],
#            '1': _1*100, 'X':_X*100, '2':_2*100,
#             '1X':(_1+_X)*100, 'X2':(_X+_2)*100, '12':(_1+_2)*100}
    
    probs = {'Match': match_outcome.Match[0],
            '1': _1, 'X':_X, '2':_2,
             '1X':(_1+_X), 'X2':(_X+_2), '12':(_1+_2)}
    
#    df_probs = pd.DataFrame(probs, index=[1]).rename(index={1:'My_Prob'})
    df_probs = pd.DataFrame(probs, index=[1])
    df_probs.insert(0, column='Info', value='My_Prob')
    
    odds = {'Match': match_outcome.Match[0],
            '1': 1/_1, 'X':1/_X, '2':1/_2,
             '1X':1/(_1+_X), 'X2':1/(_X+_2), '12':1/(_1+_2)}
    
#    my_odds = pd.DataFrame(odds, index=[1]).rename(index={1:'My_Odds'})
    my_odds = pd.DataFrame(odds, index=[1])
    my_odds.insert(0, column='Info', value='My_Odds')
    
    
    df_result = df_probs.append(my_odds, sort=False)
    
    return df_result.round(2)

def getAll1X2(test, model, save=True):
    nRound = test.iloc[0].Round
    df_result = pd.DataFrame()
    
    for i in range(len(test)):
            
        team = test.iloc[i].HomeTeam
        opponent = test.iloc[i].AwayTeam    
        
        outcome = model.simulate_match(team, opponent)
        res = get1X2(outcome)
        
        df_result = df_result.append(res, sort=False)
        
    df_result = divideDataset(df_result)
    
    if(save):
        filename = result_path + model.league_name + '/Round_' + str(nRound) + '_1X2' + '.xlsx'
        df_result.to_excel(filename)
    
    return df_result 

def getOvUnGoalNoGoal(match_outcome, max_goals = 8):

    match = match_outcome.Match
    outcome = match_outcome.Outcome.values[0]
#    print(outcome)
#    return outcome
    adj_outcome = np.rot90(outcome, axes=(1,0))
    
    # PROBABILITIES
    
    no_goal = outcome[0][0] + outcome[1][0] + outcome[0][1]
    goal = 1-no_goal
    
    under1_5 = np.sum(np.triu(adj_outcome, k=7))
    under2_5 = np.sum(np.triu(adj_outcome, k=6))
    under3_5 = np.sum(np.triu(adj_outcome, k=5))
    
    over1_5 = np.sum(np.tril(adj_outcome, k=6))
    over2_5 = np.sum(np.tril(adj_outcome, k=5))
    over3_5 = np.sum(np.tril(adj_outcome, k=4))
    
#    under1_5 = 1-over1_5
#    under2_5 = 1-over2_5
#    under3_5 = 1-over3_5
    
    df_probs = pd.DataFrame({'Match': match,
                          'Goal': goal, 'No_Goal':no_goal,
                          'Over_1.5':over1_5, 'Over_2.5':over2_5, 'Over_3.5':over3_5,
                          'Under_1.5':under1_5, 'Under_2.5':under2_5, 'Under_3.5':under3_5})
    
    df_probs.insert(0, column='Info', value='Prob')
        
    # ODDS
    no_goal = 1/no_goal
    goal = 1/goal
    over1_5 = 1/over1_5
    over2_5 = 1/over2_5
    over3_5 = 1/over3_5
    under1_5 = 1/under1_5
    under2_5 = 1/under2_5
    under3_5 = 1/under3_5
    
    df_odds = pd.DataFrame({'Match': match,
                            'Goal': goal, 'No_Goal':no_goal,
                          'Over_1.5':over1_5, 'Over_2.5':over2_5, 'Over_3.5':over3_5,
                          'Under_1.5':under1_5, 'Under_2.5':under2_5, 'Under_3.5':under3_5})
    
    df_odds.insert(0, column='Info', value='Odds')        
        
#    df_result = divideData(df_probs, df_odds)
    df_result = df_probs.append(df_odds, sort=False)
    
    return df_result
    
def getAllOvUnGoalNoGoal(test, model, save=True):
    
    df_result = pd.DataFrame()
    nRound = test.iloc[0].Round
    for i in range(len(test)):
            
        team = test.iloc[i].HomeTeam
        opponent = test.iloc[i].AwayTeam    
        
        outcome = model.simulate_match(team, opponent)
        res = getOvUnGoalNoGoal(outcome)
        
        df_result = df_result.append(res, sort=False).round(2)
    
    df_result = divideDataset(df_result)
    
    if(save):
        filename = result_path + model.league_name + '/Round_' + str(nRound) + '_OV-UN' + '.xlsx'
        df_result.to_excel(filename, index=False)
        
    return df_result
    
def getMultigoal(match_outcome, min_goal, max_goal):
    
    goals_list = [i for i in range(min_goal, max_goal+1)]
    
    match = match_outcome.Match[0]
    outcome = match_outcome.Outcome.values[0]

    probs = []
    
    for goals in goals_list:
        i=0
        
        while(goals - i >= 0):
            probs.append(outcome[goals - i][i])
            i += 1
    
    prob = np.sum(probs)
    odd = 1/prob
    
    result = pd.DataFrame({'Match':match,
                           'Prob': prob, 'Odd': odd}, index=[1])

#    result = result.rename(index={1:'Multigoal_{}-{}'.format(min_goal, max_goal)})
    result.insert(0, column='Info', value='Mul_{}-{}'.format(min_goal, max_goal))
    
    return result

def getMultigoalTeam(match_outcome, min_goal, max_goal, teamField):
    
    if(teamField != 'home' and teamField != 'away'):
        raise Exception('MultigoalTeam requires home or away label as TEAM_FIELD')
    
    goals_list = [i for i in range(min_goal, max_goal+1)]
    
    match = match_outcome.Match[0]
    outcome = match_outcome.Outcome.values[0]
    
    probs = []
    
    for goals in goals_list:
        if(teamField == 'home'):
            probs.append(np.sum(outcome[goals]))
        else:
            probs.append(np.sum(outcome[:,goals]))
    
    prob = np.sum(probs)
    odd = 1/prob
    
    if(teamField == 'home'):
        team = match.split()[0]
    elif(teamField == 'away'):
        team = match.split()[2]
    
    result = pd.DataFrame({'Match':team,
                           'Prob': prob, 'Odd': odd}, index=[1])

#    result = result.rename(index={1:'{}_Mul_{}-{}'.format(team, min_goal, max_goal)})
    result.insert(0, column='Info', value='{}_Mul_{}-{}'.format(team, min_goal, max_goal))
    
    return result

def getAllMultigoal(test, model, save=True):
    
    df_result = pd.DataFrame()
    nRound = test.iloc[0].Round
    
    for i in range(len(test)):
            
        team = test.iloc[i].HomeTeam
        opponent = test.iloc[i].AwayTeam    
        
        outcome = model.simulate_match(team, opponent)
        
        mul_13 = getMultigoal(outcome, 1,3)
        mul_14 = getMultigoal(outcome, 1,4)
        mul_24 = getMultigoal(outcome, 2,4)
        mul_25 = getMultigoal(outcome, 2,5)
        home_mul_13 = getMultigoalTeam(outcome, 1,3, 'home')
        home_mul_14 = getMultigoalTeam(outcome, 1,4, 'home')
        home_mul_24 = getMultigoalTeam(outcome, 2,4, 'home')
        home_mul_25 = getMultigoalTeam(outcome, 2,5, 'home')
        away_mul_13 = getMultigoalTeam(outcome, 1,3, 'away')
        away_mul_14 = getMultigoalTeam(outcome, 1,4, 'away')
        away_mul_24 = getMultigoalTeam(outcome, 2,4, 'away')
        away_mul_25 = getMultigoalTeam(outcome, 2,5, 'away')
        
        df_result = df_result.append(mul_13, sort=False)
        df_result = df_result.append(mul_14, sort=False)
        df_result = df_result.append(mul_24, sort=False)
        df_result = df_result.append(mul_25, sort=False)
        df_result = df_result.append(home_mul_13, sort=False)
        df_result = df_result.append(home_mul_14, sort=False)
        df_result = df_result.append(home_mul_24, sort=False)
        df_result = df_result.append(home_mul_25, sort=False)
        df_result = df_result.append(away_mul_13, sort=False)
        df_result = df_result.append(away_mul_14, sort=False)
        df_result = df_result.append(away_mul_24, sort=False)
        df_result = df_result.append(away_mul_25, sort=False)
        
        df_result = df_result.round(2)
#        return df_result
    df_result = divideDataset(df_result, gap=4)
        
    if(save):
        filename = result_path + model.league_name + '/Round_' + str(nRound) + '_Multigoals' + '.xlsx'
        df_result.to_excel(filename, index=False)
        
    return df_result
    
    

def getAllBookmakersOdds(test):
    
    df_result = pd.DataFrame()
    
    for i in range(len(test)):
            
        team = test.iloc[i].HomeTeam
        opponent = test.iloc[i].AwayTeam    
        
        # PROBABILITIES
        bet_1 = test.iloc[i].Bet_1
        bet_X = test.iloc[i].Bet_X
        bet_2 = test.iloc[i].Bet_2
        
        bet_1X = 1/bet_1 + 1/bet_X
        bet_X2 = 1/bet_2 + 1/bet_X
        bet_12 = 1/bet_1 + 1/bet_2
#        bet_1 = 100*(1/bet_1) / tot_bet_prob
#        bet_X = 100*(1/bet_X) / tot_bet_prob
#        bet_2 = 100*(1/bet_2) / tot_bet_prob
        bet_1 = (1/bet_1)
        bet_X = (1/bet_X)
        bet_2 = (1/bet_2)        
        
        probs = {'Match': str(team + ' vs ' + opponent),
                '1':bet_1, 'X':bet_X, '2':bet_2,
                '1X':bet_1X, 'X2':bet_X2, '12':bet_12}
        
#        df_probs = pd.DataFrame(probs, index=[1]).rename(index={1:'bet365_probs'})
        df_probs = pd.DataFrame(probs, index=[1])
        df_probs.insert(0, column='Info', value='bet365_Probs')
        
        # ODDS
        bet_1 = test.iloc[i].Bet_1
        bet_X = test.iloc[i].Bet_X
        bet_2 = test.iloc[i].Bet_2
        bet_1X = 1/(1/bet_1 + 1/bet_X)
        bet_X2 = 1/(1/bet_X + 1/bet_2)
        bet_12 = 1/(1/bet_1 + 1/bet_2)
        
        odds = {'Match': str(team + ' vs ' + opponent),
                '1':bet_1, 'X':bet_X, '2':bet_2,
                '1X':bet_1X, 'X2':bet_X2, '12':bet_12}
        
#        df_odds = pd.DataFrame(odds, index=[1]).rename(index={1:'bet365_odds'})
        df_odds = pd.DataFrame(odds, index=[1])
        df_odds.insert(0, column='Info', value='bet365_Probs')
        
        df_result = df_result.append(df_probs, sort=False)
        df_result = df_result.append(df_odds, sort=False)
        
    return df_result.round(2)

def basicSelection(basic_df, prob, col):
    
    basic_df = basic_df.reset_index()
    
    prob_df = basic_df.loc[basic_df['Info']=='My_Prob']
    prob_1x2 = prob_df[[col]]
    selection = prob_1x2.loc[prob_1x2[col]> prob]
    selection_indexes = selection.index
    selection_indexes
    
    sel = basic_df[['Info', 'Match', col]]
    sel = sel.loc[selection_indexes]
    
    sel['ODD_{}'.format(col)] = 1/sel[col]
    sel = sel.round(2)
    
    return sel

def unovSelection(basic_df, prob, col):
    
    basic_df = basic_df.reset_index()
    
    prob_df = basic_df.loc[basic_df['Info']=='Prob']
    prob_unov = prob_df[[col]]
    selection = prob_unov.loc[prob_unov[col]> prob]
    selection_indexes = selection.index
    selection_indexes
    
    sel = basic_df[['Info', 'Match', col]]
    sel = sel.loc[selection_indexes]
    
    sel['ODD_{}'.format(col)] = 1/sel[col]
    sel = sel.round(2)
    
    return sel

def selection(model, nRound, _1x2_df, unov_df, _12_prob, _1x_x2_prob, over1_5_prob, under3_5_prob):
    
    _1 = basicSelection(_1x2_df, _12_prob, '1')
    _2 = basicSelection(_1x2_df, _12_prob, '2')
    _1x = basicSelection(_1x2_df, _1x_x2_prob, '1X')
    _x2 = basicSelection(_1x2_df, _1x_x2_prob, 'X2')
    
    _1x2_ = _1.merge(_1x, how='outer').merge(_x2, how='outer').merge(_2, 'outer')
#    col = _1x2_.pop('Odd')
#    _1x2_.insert(6, 'Odd', col)    
    
    ov = unovSelection(unov_df, over1_5_prob, 'Over_1.5')
    un = unovSelection(unov_df, under3_5_prob, 'Under_3.5')
    
    unov = ov.merge(un, how='outer')
    
    final = _1x2_.merge(unov, how='outer')
    
    filename = result_path + model.league_name + '/Round_' + str(nRound) + '_Selection' + '.xlsx'
    final.to_excel(filename, index=False)
    
    return final
    
#%% COLOR STYLE        
        
def highlight_cells(color):
    return ['background-color: ' + color]
        
#df.style.apply(highlight_cells)        
        
def color(color):
    color = 'red'
    return 'background-color: %s' % color
        
        
        