# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:25:45 2016

@author: iralp
"""
import pandas as pd
import numpy as np

columns_bat = None
columns_pitch = None
playerstats_bat = None
playerstats_pitch = None

def populateBatStats(bat_stats,reqColumns):
    global playerstats_bat
    cols = bat_stats.columns
    for each_col in bat_stats.columns:
        cols.append(each_col + "_min")
        cols.append(each_col + "_max")
        cols.append(each_col + "_mean")
        cols.append(each_col + "_var")
        
    playerstats_bat = pd.DataFrame()
    playerstats_bat[bat_stats.columns] = bat_stats[bat_stats.columns]
    players = distinct(bat_stats['player_ID'])
    #for each player,get the stats for each distinct year
    for each_player in players:
        ages = sorted(bat_stats[bat_stats['player_ID'] == each_player]['age'])
        for age in ages:
            for each_column in reqColumns:
                indx = bat_stats[bat_stats['player_ID'] == each_player][bat_stats['age'] == 'age'].index
                col_value =  bat_stats[bat_stats['player_ID'] == each_player][bat_stats['age'] <= 'age'][each_column]
                playerstats_bat.iloc[indx][each_column + "_min"] = min(col_value)
                playerstats_bat.iloc[indx][each_column + "_max"] = max(col_value)
                playerstats_bat.iloc[indx][each_column + "_mean"] = np.mean(col_value)
                playerstats_bat.iloc[indx][each_column + "_var"] = np.var(col_value)
                
            
        
#compute the mean,variance,min max for the players by the end of each year  
def Calcstats():
    #columns = ['playerID','player_name','age','year','mean','variance','min','max']
    #stats = pd.DataFrame(columns = columns)

    #Load the actual data
    bat_stats = pd.DataFrame().from_csv(r'C:/Personal_Docs/Course_Work/Data Science/Course_Project/Data/war_daily_bat.csv',encoding='utf8',index_col = False)
    pitch_stats = pd.DataFrame().from_csv(r'C:/Personal_Docs/Course_Work/Data Science/Course_Project/Data/war_daily_pitch.csv',encoding='utf8',index_col = False)
    
    print(bat_stats.columns)
    print(pitch_stats.columns)
    
    #Normalize the required columns for batters
    reqColumns = list(bat_stats.columns)
    reqColumns.remove('name_common')
    reqColumns.remove('age')
    reqColumns.remove('mlb_ID')
    reqColumns.remove('player_ID')
    reqColumns.remove('year_ID')
    reqColumns.remove('team_ID')
    reqColumns.remove('stint_ID')
    reqColumns.remove('lg_ID')
    reqColumns.remove('salary')    
    reqColumns.remove('pitcher')
    
    bat_cal_stats = populateStats(bat_stats,reqColumns)
    
    
    
    
    
Calcstats()