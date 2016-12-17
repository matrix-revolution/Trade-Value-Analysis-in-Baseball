# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 22:34:00 2016

@author: iralp
"""
import pandas as pd
from copy import deepcopy 
from sklearn import preprocessing
import warnings
from scipy import spatial 
import numpy as np
import operator
from sklearn import decomposition

warnings.filterwarnings("ignore")
#Data Science Project
#For a given pair of players,get the most comparable players

playerDataBat = None
playerDataPitch = None

def normalizeColumns(playerDataBatCopy,reqColumns):
    for each_column in reqColumns:
        #replace invalid values with valid ones
        for indx in range(len(playerDataBatCopy)):
            if playerDataBatCopy.iloc[indx][each_column] == 'NULL':
                playerDataBatCopy.iloc[indx][each_column] = 0
        #processing for Nans
        playerDataBatCopy[each_column] = playerDataBatCopy[each_column].fillna(value = 0) 
        print('column  :',each_column)
        #preprocessing.normalize(playerDataBatCopy[each_column],axis = 0)
        #print(" Values : ",len(playerDataBatCopy[each_column])) 
        #playerDataBatCopy[each_column] = preprocessing.normalize(playerDataBatCopy[each_column],axis = 0)
        print(len(playerDataBatCopy[each_column]))
    print('Normalization Done!')

    
def scoreWithBatters(playerID,playerDataBatCopy,reqColumns,year_of_trade):
    #count = 0
    #sum_ = 0
    reqColumns = ['WAR']
    #, 5, 6, 7, 8, 9
    dict_of_scores = {}
    print('player1 : ',playerID)
    #get the player data
    player1_ = playerDataBatCopy[playerDataBatCopy['player_ID'] == playerID]
    player1_data = player1_[player1_['year_ID'] <= year_of_trade]
    print('player1_data : ',len(player1_data))
    #print('player1_data : ',player1_data)
    
    #sort by age
    player1_data = player1_data.sort(['age'])
    #for each year,get the corresponding similarity score
    for indx in range(len(player1_data)):
        #get the age
        age = player1_data.iloc[indx]['age']
        vec1 = player1_data.iloc[indx][reqColumns]
        #print('age : ',age) 
        
        #get rows with same age
        rows_ = playerDataBatCopy[playerDataBatCopy['age'] == age]
        rows_with_age = rows_[rows_['player_ID'] != playerID]
        
        for index,row in rows_with_age.iterrows():
            #print('row : ',row)
            name = row['name_common']
            #print('name : ',name,' age:',age,' WAR : ',row['WAR'])
            vec2 = row[reqColumns]
            #print('vec1 :',np.array(vec1))
            #print('vec2: ',np.array(vec2))
            score = 1 - spatial.distance.cosine(np.array(vec1),np.array(vec2))
            #print('score : ',score)
            if name not in dict_of_scores.keys():
                dict_of_scores[name] = (score,1)
            else:
                tup = dict_of_scores[name]
                no = tup[1] + 1
                sco = (tup[0] + score)
                dict_of_scores[name] = (sco,no)

    for each_key in dict_of_scores:
        tp = dict_of_scores[each_key]
        dict_of_scores[each_key] = ((tp[0]/tp[1]),no)
 
    #return the top ten most comparable players
    sorted_dict = sorted(dict_of_scores.items(),key = operator.itemgetter(1),reverse = True)
    return sorted_dict[:10]    
    
def getNextWAR(playerID,pl1_comparables,playerDataBatCopy,year_of_trade):
    #get the current age of player1
    player1_ = playerDataBatCopy[playerDataBatCopy['player_ID'] == playerID]
    player1_data = player1_[player1_['year_ID'] <= year_of_trade]
  
    #sort by age
    player1_data = player1_data.sort(['age'],ascending=False)
    if len(player1_data) == 0:
        print('No comparables')
        return
        
    age_p1 = player1_data.iloc[0]['age']
    WAR_years = []
    #print('comparables : ',pl1_comparables)
    #average the WAR for this for all the 10 most comparables
    for ind in range(1,4):        
        cosine_sum = 0         
        next_age = age_p1 + ind
        WAR_ = 0.0
        count = 0
 
        for indx in range(len(pl1_comparables)):    

            each_player = pl1_comparables[indx][0]
            #print('each_player : ',each_player)
            player_ = playerDataBatCopy[playerDataBatCopy['name_common'] == each_player]
            player_data = player_[player_['age'] == next_age]
            if len(player_data) == 0:
                continue
            
            print('player_data[WAR] : ',player_data['copied_WAR'])
            WAR_ += (player_data.iloc[0]['copied_WAR'])
            cosine_sum += pl1_comparables[indx][1][0]
            count += 1
            print('WAR_predict : ',WAR_)
        if cosine_sum == 0:
            cosine_sum = 1
        if count == 0:
            count = 1
        WAR_ = WAR_/count
        WAR_years.append(WAR_)
        
    return WAR_years
    
def getComparablePlayers(playerID,year_of_trade):
    #Use all features and compute some similarity score after normalizing the features=>
    #but consider only data corresponding to the same age
    global playerDataBat
    global playerDataPitch
    #load all the data
    playerDataBat = pd.DataFrame().from_csv(r'C:/Personal_Docs/Course_Work/Data Science/Course_Project/Data/player_daily_bat_Normalized.csv',encoding='utf8',index_col = False)
    playerDataPitch = pd.DataFrame().from_csv(r'C:/Personal_Docs/Course_Work/Data Science/Course_Project/Data/player_daily_pitch_Normalized.csv',encoding='utf8',index_col = False)
    playerDataBatCopy = deepcopy(playerDataBat)
    playerDataPitchCopy = deepcopy(playerDataPitch)  
    
    playerID = str(playerID).strip()
    print('get comparable players for : ',playerID)
    
    #Now determine if either player is a hitter or a pitcher
    player1_rows = playerDataBat[playerDataBat['player_ID'] == playerID]
    
    print('player1 length : ',len(player1_rows))
    
    if len(player1_rows) == 0:
        player1_rows = playerDataPitch[playerDataPitch['player_ID'] == playerID]
    
    print('player1 length : ',len(player1_rows))
    
    #Normalize the required columns for batters
    reqColumns = list(playerDataBat.columns)
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
    reqColumns.remove('teamRpG')      
    reqColumns.remove('oppRpG')
    reqColumns.remove('oppRpPA_rep')    
    reqColumns.remove('oppRpG_rep')
    reqColumns.remove('pyth_exponent')    
    reqColumns.remove('pyth_exponent_rep')
    
    #Normalize the required columns for batters
    reqColumns_pitchers = list(playerDataPitchCopy.columns)
    reqColumns_pitchers.remove('name_common')
    reqColumns_pitchers.remove('age')
    reqColumns_pitchers.remove('mlb_ID')
    reqColumns_pitchers.remove('player_ID')
    reqColumns_pitchers.remove('year_ID')
    reqColumns_pitchers.remove('team_ID')
    reqColumns_pitchers.remove('stint_ID')
    reqColumns_pitchers.remove('lg_ID')
    reqColumns_pitchers.remove('salary')    
   
    playerDataBatCopy['copied_WAR'] = playerDataBatCopy['WAR']
    playerDataPitchCopy['copied_WAR'] = playerDataPitchCopy['WAR']
    normalizeColumns(playerDataBatCopy,['WAR'])
    normalizeColumns(playerDataPitchCopy,['WAR'])
    '''matrix_of_batters = np.matrix(playerDataBatCopy[reqColumns])
    pca = decomposition.PCA(n_components = 5)
    pca_fit_batters = pca.fit_transform(matrix_of_batters)
    print(pca_fit_batters.shape)
    
    matrix_of_pitchers = np.matrix(playerDataPitchCopy[reqColumns_pitchers])
    pca = decomposition.PCA(n_components = 5)
    pca_fit_pitchers = pca.fit_transform(matrix_of_pitchers)
    print(pca_fit_pitchers.shape)
    
    pitchers_frame = pd.DataFrame(pca_fit_pitchers)
    pitchers_frame['age'] = playerDataPitchCopy['age']
    pitchers_frame['name_common'] = playerDataPitchCopy['name_common']
    pitchers_frame['player_ID'] = playerDataPitchCopy['player_ID']
    pitchers_frame['year_ID'] = playerDataPitchCopy['year_ID']
    print(pitchers_frame.columns)
    
    
    batters_frame = pd.DataFrame(pca_fit_batters)
    batters_frame['age'] = playerDataBatCopy['age']
    batters_frame['name_common'] = playerDataBatCopy['name_common']
    batters_frame['player_ID'] = playerDataBatCopy['player_ID']
    batters_frame['year_ID'] = playerDataBatCopy['year_ID']
    print(batters_frame.columns)
    '''
    
    pl1_comparables = None
    pl1_WAR = None
    
    #check if the player is a pitcher or a batter
    if len(player1_rows) == 0:
        return None
    player1IsPitcher = player1_rows.iloc[0]['pitcher']
    
    #if batter consider the rows corresponding to war_bat.csv
    if player1IsPitcher == 'N':
        pl1_comparables = scoreWithBatters(playerID,playerDataBatCopy,reqColumns,year_of_trade)
        pl1_WAR = getNextWAR(playerID,pl1_comparables,playerDataBatCopy,year_of_trade)              
    else:
        pl1_comparables = scoreWithBatters(playerID,playerDataPitchCopy,reqColumns_pitchers,year_of_trade)
        pl1_WAR = getNextWAR(playerID,pl1_comparables,playerDataPitchCopy,year_of_trade)
     
    print('pl1_WAR : ',pl1_WAR)
    return pl1_WAR

#averaged over 3 years
def getActualWAR(playerID,year):
    playerID = str(playerID).strip()
    print('get comparable players for : ',playerID)
    
    #Now determine if either player is a hitter or a pitcher
    player1_rows = playerDataBat[playerDataBat['player_ID'] == playerID]
    
    print('player1 length : ',len(player1_rows))
    
    if len(player1_rows) == 0:
        player1_rows = playerDataPitch[playerDataPitch['player_ID'] == playerID]
    
    print('player1 length : ',len(player1_rows))
    
    actual_WAR = []
    #Now get the average WAR over next 3 years
    for ind in range(1,4):
        print('n th  year : ',ind)
        yr = year + ind
        row = player1_rows[player1_rows['year_ID'] == yr]
        if len(row) == 0:
            actual_WAR.append(0.0)
            continue
        WAR = row['WAR']
        WAR = str(WAR).split()
        actual_WAR.append(float(WAR[1]))
        
    return actual_WAR
    
def rmse(predictions, targets):
    return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())
    
    
def getrmse(players_results,predict,actual):
    predicted = []
    targets = []
   
    for indx in range(len(players_results)):
        if players_results.iloc[indx][predict] != None and players_results.iloc[indx][predict] != 'nan':
            try:
                predicted.append(float(players_results.iloc[indx][predict]))
            except :
                predicted.append(0.0)
        else:
            predicted.append(0.0)
            
        if players_results.iloc[indx][actual] != None and players_results.iloc[indx][predict] != 'nan':    
            try:
                targets.append(float(players_results.iloc[indx][actual]))
            except:
                targets.append(0.0)
        else:
            targets.append(0.0)

    
    print('players_results predict: ',players_results[predict])
    print('players_results actual: ',players_results[actual])
    
    rmse_score = rmse(np.array(predicted),np.array(targets))
    return rmse_score
                        
def validatePreviousData():
    players_results = pd.DataFrame(columns = ['playerID','player_name','predicted WAR_1',\
    'actual WAR_1','predicted WAR_2','actual WAR_2','predicted WAR_3','actual WAR_3',\
    'start_year','trade_year','from_team','to_team','transaction-ID','rmse_scores_1',\
    'rmse_scores_2','rmse_scores_3'])
    columns = ['primary-date',  'time', 'approximate-indicator',  'secondary-date',  'approximate-indicator (for secondary-date)',\
               'transaction-ID',  'player', 'type',  'from-team',  'from-league',  \
               'to-team',  'to-league','draft-type',  \
               'draft-round', 'pick-number',  'info']

    file_count = 0
   #load the csv into a data frame
    data = pd.read_csv('C:/Personal_Docs/Course_Work/Data Science/Course_Project/Data/transactions/tran.txt',header = 0,\
       parse_dates = True,names = columns)
    #data cleaning 
    data['time'].fillna(0,inplace = True)
    data['approximate-indicator'].fillna(0,inplace = True)
    data['secondary-date'].fillna(0,inplace = True)
    data['approximate-indicator (for secondary-date)'].fillna(0,inplace = True)
    data['draft-round'].fillna("NA",inplace = True)
    data['info'].fillna("NA",inplace = True)
    data['pick-number'].fillna("NA",inplace = True)
    data['draft-type'].fillna("NA",inplace = True)
    
    years = [int(str(x)[:4]) for x in data['primary-date']]
    months = [int(str(x)[4:6]) for x in data['primary-date']]
    
    #create a new column with year,month of trade
    data['year'] = years
    data['month'] = months 

    #load the master data to get player name from his id
    master_data = pd.read_csv('C:/Personal_Docs/Course_Work/Data Science/Course_Project/Data/baseballdatabank-master/core/Master.csv',\
        index_col = False)
    #usecols = ['retroID','nameGiven']
    
    #consider trades in ranges 2005-2010
    data_temp = data[data['year'] >= 2010]
    data_trades_05_10 = data_temp[data_temp['year'] <= 2012]
    
    transactions = set(data_trades_05_10['transaction-ID'])
    count = 0
    
    for indx,row in data_trades_05_10.iterrows():
        #for each trade,get the players => calculate their WAR for next 1-3 years and validate with original WAR
        #for each transaction id get the players involved
            for each_trade in transactions:
                #get the cooresponding rows
                trades = data_trades_05_10[data_trades_05_10['transaction-ID'] == each_trade]
                #check the trade type and skip if not T
                if trades.iloc[0]['type'] != 'T ':
                    continue
                #get the players
                
                for indx,each_ in trades.iterrows():
                    playerID = each_['player']
                    #get the player name from 
                    print('playerID: ',playerID)
                    player_data_indx = master_data[master_data['retroID'] == playerID].index
                    if player_data_indx == None:
                        continue
                    print('player_data : ',player_data_indx)
                    playerName = master_data.iloc[player_data_indx]['playerID']
                    print('playerName : ',playerName)
                    playr_ID = str(playerName).split()
                    #TODO get the comparable players also
                    player_WAR = getComparablePlayers(playr_ID[1],each_['year'])
                    
                    if player_WAR == None:
                        continue
                    
                    #computer the actual WAR average for next three years
                    actual_WAR = getActualWAR(playr_ID[1],each_['year'])
                    
                    player_result = []
                    player_result.append(playr_ID[1] + " ")
                    player_result.append(master_data.iloc[player_data_indx]['nameGiven'])
                    player_result.append(str(player_WAR[0]) + "")
                    player_result.append(str(actual_WAR[0]) + "")
                    player_result.append(str(player_WAR[1]) + "")
                    player_result.append(str(actual_WAR[1]) + "")
                    player_result.append(str(player_WAR[2]) + "")
                    player_result.append(str(actual_WAR[2]) + "")
                    
                    player_result.append(str(int(each_['year']) + 1)  + " ")
                    player_result.append(each_['year'])
                    player_result.append(each_['from-team'])
                    player_result.append(each_['to-team'])
                    player_result.append(each_trade)
                    player_result.append("0")
                    player_result.append("0")
                    player_result.append("0")
                    
                    players_results.loc[count] = player_result
                    
                    count += 1
                    print('length of frame :',len(players_results))
                    if count == 100: 
                        rmse_score_1 = getrmse(players_results,'predicted WAR_1','actual WAR_1')
                        rmse_score_2 = getrmse(players_results,'predicted WAR_2','actual WAR_2')
                        rmse_score_3 = getrmse(players_results,'predicted WAR_3','actual WAR_3')
                        players_results['rmse_scores_1'] = [rmse_score_1]*100
                        players_results['rmse_scores_2'] = [rmse_score_2]*100
                        players_results['rmse_scores_3'] = [rmse_score_3]*100

                        players_results.to_csv('C:/Personal_Docs/Course_Work/Data Science/Course_Project/Data/Results_'+str(file_count)+'.csv',encoding = 'utf8')
                        count = 0
                        file_count += 1
                        
                        
                        
    return players_results
                    
data_results = validatePreviousData()
#data_results.to_csv('C:/Personal_Docs/Course_Work/Data Science/Course_Project/Data/Results_5.csv',encoding = 'utf8')
print('Done!!')
