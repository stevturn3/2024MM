import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from os.path import join
##########GLOBALS##########
classifiers = {
    'RF': RandomForestClassifier(),
    'SVC' : SVC(),
    'XG': XGBClassifier()
}

cv_methods = {
    'RF' : {
    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features' : ['sqrt', 'log2']
},
    'SVC' : {
    'C': [1, 10],
    'gamma': [1, 0.1],
    'kernel': ['rbf', 'poly']
},
    'XG' : {
    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    'learning_rate': [0.01, 0.1, 0.3, 0.5],
    'subsample': [0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
    'gamma': [0, 1, 5],
    'reg_lambda': [0, 1, 5],
    'reg_alpha': [0, 1, 5],
    'min_child_weight': [1, 3, 5, 7]
}
}
score = {1: 3, 2: 2, 3:1}
##########INGEST##########
def load_df(df_name, dir, describe=False):
    """
    Load a dataframe from a CSV file.

    Args:
    - df_name (str): The name of the CSV file (without the extension).
    - dir (str): The directory where the CSV file is located.
    - describe (bool): Whether to display descriptive statistics of the loaded dataframe.

    Returns:
    - pd.DataFrame: The loaded dataframe.
    """
    df = pd.read_csv(join(dir,f"{df_name}.csv"), encoding='cp1252') # Load df into polars
    if describe:
        display(df.describe())
    return df
##########Transform##########
def lloc(x):
    """
    Determine the 'LLoc' based on the 'WLoc' column value.

    Args:
    - x (pd.Series): A row from a dataframe containing the 'WLoc' column.

    Returns:
    - str: The location ('H', 'A', or 'N').
    """
    if x['WLoc'] == "H":  
        return "A"
    elif x['WLoc'] == "A": 
        return "H"
    else:
        return "N"  

def q(loc, rpi):
    """
    Assign a quadrant value based on RPI, and location.

    Args:
    - loc (str): Location ('H', 'A', or 'N').
    - rpi (int): RPI value.

    Returns:
    - int: Quadrant value (1, 2, or 3).
    """
    if 1 <= rpi <= 30 and loc == "H":
        return 1
    elif 31 <= rpi <= 75 and loc == "H":
        return 2
    elif 1 <= rpi <= 50  and loc == "N":
        return 1
    elif 51 <= rpi <= 100 and loc == "N":
        return 2
    elif 1 <= rpi <= 75 and loc == "A":
        return 1
    elif 76 <= rpi <= 135 and loc == "A":
        return 2
    else:
        return 3
    
def establish_quad(x):
    """
    Establish quadrants for winning and losing teams based on RPI and location.

    Args:
    - x (pd.Series): A row from a dataframe containing relevant columns.

    Returns:
    - tuple: Quadrant values for the winning and losing teams. Quadrant corresponds
    to the quadrant of the opposite team.
    """
    wid = x["WTeamID"]
    lid = x["LTeamID"]
    wloc = x["WLoc"]
    lloc = None
    if wloc == "H":  
        lloc = "A"
    elif (wloc == "A"): 
        lloc = "H"
    else:
        lloc = "N" 
    wrpi = x["Wrpi"]
    lrpi = x["Lrpi"]
    w_q = q(lloc, lrpi)
    l_q = q(wloc, wrpi)
    return w_q, l_q  

def gather_dat(mMassey, mRegSeasonResults):
    """
    Combine Massey and regular season results dataframes. Performs feature engineering to provide
    defensive and pace statistics.

    Args:
    - mMassey (pd.DataFrame): Massey Indicators dataframe.
    - mRegSeasonResults (pd.DataFrame): Regular season results dataframe.

    Returns:
    - pd.DataFrame: Data frame with RPI, regular season statistics, and additional features.
    """
    mRegSeasonResults["LLoc"] = mRegSeasonResults.apply(lloc, axis = 1)
    mr_temp = mRegSeasonResults
    if mMassey is not None:
        rpi = mMassey.loc[mMassey['SystemName'] == "RPI",]
        idx = rpi.groupby(['Season'])['RankingDayNum'].transform(max) == rpi['RankingDayNum']
        rpi = rpi[idx]
        mr_temp = pd.merge(mRegSeasonResults, rpi, how = 'left', left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID']).drop(['TeamID', 'SystemName','RankingDayNum'], axis=1)
        t = list(mr_temp.columns)
        t[-1] = "Wrpi"
        mr_temp.columns = t
        mr_temp = pd.merge(mr_temp, rpi, how = 'left', left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID']).drop(['TeamID', 'SystemName','RankingDayNum'], axis=1)
        t = list(mr_temp.columns)
        t[-1] = "Lrpi"
        mr_temp.columns = t
    mr_temp['WFGP'] = mr_temp['WFGM'] / mr_temp['WFGA']
    mr_temp['LFGP'] = mr_temp['LFGM'] / mr_temp['LFGA']
    mr_temp['WFGP3'] = mr_temp['WFGM3'] / mr_temp['WFGA3']
    mr_temp['LFGP3'] = mr_temp['LFGM3'] / mr_temp['LFGA3']
    mr_temp['ScoreGap'] = mr_temp['WScore'] - mr_temp['LScore']
    mr_temp['WPace'] = mr_temp['WFGA'] + mr_temp['WFTA']/2 + mr_temp['WTO'] - mr_temp['WOR']
    mr_temp['LPace'] = mr_temp['LFGA'] + mr_temp['LFTA']/2 + mr_temp['LTO'] - mr_temp['LOR']
    if mMassey is not None:
        res = mr_temp.apply(establish_quad, axis = 1)
        mr_temp["WQuad"] = res.apply(lambda x: x[0])
        mr_temp["LQuad"] = res.apply(lambda x: x[1])
    mr_temp['WLoc'] = mr_temp['WLoc'].map({'H':1,'N':2, 'A':3})
    mr_temp['LLoc'] = mr_temp['LLoc'].map({'H':1,'N':2, 'A':3})
    mr_temp['WOPFGP'] =  mr_temp['LFGP']
    mr_temp['LOPFGP'] =  mr_temp['WFGP'] 
    mr_temp['WOPFGP3'] =  mr_temp['LFGP3'] 
    mr_temp['LOPFGP3'] =  mr_temp['WFGP3']
    if mMassey is not None:
        mr_temp['Wrpi']=mr_temp['Wrpi'].fillna(0)
        mr_temp['Lrpi']=mr_temp['Lrpi'].fillna(0)
    wins = mr_temp[[col for col in mr_temp.columns if col.startswith('W')]]
    loss = mr_temp[[col for col in mr_temp.columns if col.startswith('L')]]
    wins['Season'] = mr_temp['Season']
    wins['WTeamID'] = mr_temp['WTeamID']
    wins['win'] = 1
    loss['Season'] = mr_temp['Season']
    loss['LTeamID'] = mr_temp['LTeamID']
    loss['win'] = 0
    wins.columns = [col.replace("W", "") for col in wins.columns]
    loss.columns = [x if x != "oc" else "Loc" for x in [col.replace("L", "") for col in loss.columns]]
    res = pd.concat([wins, loss])
    if mMassey is not None:
        res = res.groupby(['Season', 'TeamID', 'Quad'], as_index = False).agg('mean')
    else:
        res = res.groupby(['Season', 'TeamID'], as_index = False).agg('mean')
    return res

def build_data(dat, team_ID, season, quad_weight = False, drop_c = ['win']):
    """
    Build team data for a specific season.

    Args:
    - dat (pd.DataFrame): The compiled data containing team statistics.
    - team_ID (int): The ID of the team.
    - season (int): The season year.
    - quad_weight (bool): Whether to apply quadrant weighting.
    - drop_c (list): List of columns to drop.

    Returns:
    - pd.DataFrame: Team data for the specified team and season.
    """
    team_dat = dat.loc[(dat["TeamID"] == team_ID) &  (dat["Season"] == season),].drop(drop_c,axis=1)
    if quad_weight:
        team_dat['Score'] = [score[s] for s in team_dat['Quad']]
        for idx, row in team_dat.iterrows():
            team_dat.loc[idx, 'FGM':] *= row['Score']
    team_dat = team_dat.groupby(['Season','TeamID']).agg('mean')
    return team_dat

def build_ps_data(dat_compiled, torney_dat, qw = False):
    """
    Build tournament data for prediction.

    Args:
    - dat_compiled (pd.DataFrame): Compiled team data.
    - torney_dat (pd.DataFrame): Tournament data.
    - qw (bool): Whether to apply quadrant weighting.

    Returns:
    - pd.DataFrame: Processed tournament data for prediction. One row for each game in
    tournaments from 2003 on. Each row will contain the result of the tournament game 
    and regular season summary statistics for each team. Team labels have been randomized
    to create balanced classes.
    """
    data = []
    for idx, row in torney_dat.iterrows():
        winner = None
        season = row['Season']
        team_A = [min(row['WTeamID'], row['LTeamID'])]
        team_B = [max(row['WTeamID'], row['LTeamID'])]
        if team_A[0] == row['WTeamID']:
            winner = [0]
        else:
            winner = [1]
        a_dat = build_data(dat_compiled, team_A[0], season, quad_weight= qw)
        a_col = list(a_dat.columns + "A")
        a_dat = a_dat.iloc[0,].tolist()
        b_dat = build_data(dat_compiled, team_B[0], season, quad_weight= qw)
        b_col = list(b_dat.columns + "B")
        b_dat = b_dat.iloc[0,].tolist()
        row = [season, team_A[0], team_B[0]]
        row.extend(a_dat)
        row.extend(b_dat)
        row.extend(winner)
        data.append(row)
    cols = ['Season', 'team_A', 'team_B']
    cols.extend(a_col)
    cols.extend(b_col) 
    cols.append('winner')
    data_df = pd.DataFrame(data, columns = cols)
    return data_df

##########TRAIN##########
def train_model(x_train, y_train, classifiers_l, cv_methods):
    """
    Trains machine learning models using RandomizedSearchCV.

    Parameters:
    - x_train (DataFrame): Tourney Training data, previous years data.
    - y_train (Series): Training Win or loss for each game.
    - classifiers_l (list): List of classifiers to train.
    - cv_methods (dict): Dictionary containing hyperparameter search spaces for each classifier.

    Returns:
    - fitted (list): List of best fitted models for each classifier.
    """
    fitted = []
    for l in classifiers_l:
        classifier = classifiers[l]
        CV_gs = RandomizedSearchCV(estimator=classifier, param_distributions=cv_methods[l], cv=3, n_jobs=-1, n_iter = 100)
        CV_gs.fit(x_train, y_train)
        best_mod = CV_gs.best_estimator_
        fitted.append(best_mod)
    return fitted

def test_models(fitted, x_test, y_test):
    """
    Tests fitted models on test data and calculates accuracy scores.

    Parameters:
    - fitted (list): List of fitted models to test.
    - x_test (DataFrame): Tourney Test data. 
    - y_test (Series):  Testing Win or loss for each game.

    Returns:
    - acc (list): List of accuracy scores for each fitted model.
    """
    acc = []
    for f in fitted:
        res = f.predict(x_test)
        acc.append(accuracy_score(res, y_test))
    return acc

def model_data(data):
    """
    Trains and tests models on previous seasons data.

    Parameters:
    - data (DataFrame): Full Merged Tourney data. The data should be the 
    result of build_ps_data

    Returns:
    - acc (list): List of accuracy scores for each model.
    - fitted_models (list): List of best fitted models.
    """
    X = data.drop(['winner'], axis = 1)
    y = data['winner']
    X  = X.drop(['Season', 'team_A', 'team_B'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1)
    fitted_models = train_model(X_train, y_train, ['RF', 'XG'], cv_methods)
    acc = test_models(fitted_models, X_test, y_test)
    return acc, fitted_models

def build_matchups(teams, rg_data):
    """
    Builds possible matchups DataFrame between teams in current years torunament.

    Parameters:
    - teams (DataFrame): DataFrame containing team data.
    - rg_data (DataFrame): DataFrame containing regular season summary data.

    Returns:
    - mc (DataFrame): DataFrame with team matchups in accordance with build_ps_data style.
    """
    mc = teams[['TeamID']]
    mc = pd.merge(mc, mc, how='cross')
    mc = mc[mc['TeamID_x'] != mc['TeamID_y']]
    mc = pd.merge(mc, rg_data, left_on =['TeamID_x'], right_on = ['TeamID'], how='left')
    mc = mc.drop(['TeamID'],axis=1)
    mc = pd.merge(mc, rg_data, left_on =['TeamID_y'], right_on = ['TeamID'], how='left')
    return mc

def build_prediction_dat(tourney_slots, tourney_seeds, teams, rg_data, predict_data, model, gender = 'M'):
    """
    Builds prediction data based on tournament data and models.

    Parameters:
    - tourney_slots (DataFrame): DataFrame containing tournament slot data.
    - tourney_seeds (DataFrame): DataFrame containing tournament seed data.
    - teams (DataFrame): DataFrame containing team data.
    - rg_data (DataFrame): DataFrame containing regular season summary data.
    - predict_data (DataFrame): Full predictive data.
    - model: Trained machine learning model.
    - gender (str): Gender label for the tournament data.

    Returns:
    - tourney_slots (DataFrame): Current Tournament slot data.
    - seeds (dict): Dictionary containing tournament seeds.
    - id_from_seeds (dict): Dictionary mapping seeds to IDs.
    - win_dict (dict): Dictionary containing win probabilities for each
    possible game in the current tournament.
    """
    year = np.max(tourney_slots['Season'])
    tourney_slots = tourney_slots[tourney_slots['Season'] == year]
    tourney_slots = tourney_slots[tourney_slots['Slot'].str.contains('R')] 
    seeds = tourney_seeds[tourney_seeds['Tournament'] == gender].set_index('Seed')['TeamID'].to_dict()
    id_from_seeds = {value : key for key,value in seeds.items()}
    tms = np.unique(list(seeds.values()))
    teams = teams[[True if t in tms else False for t in teams['TeamID']]]

    mc = build_matchups(teams, rg_data)
    t = mc.columns
    t = [x.replace('_x', 'A') for x in t ]
    t = [x.replace('_y', 'B') for x in t ]
    mc.columns = t
    teamA = mc['TeamIDA']
    teamB = mc['TeamIDB']
    mc = mc.loc[:, [t in predict_data.columns for t in mc.columns]]
    mc.fillna(2, inplace=True)
    probwin = model.predict_proba(mc)[:,1]
    #Prob teamB wins:
    win_dict = {(a, b) : c for a,b,c in zip(teamA, teamB, probwin)}
    return tourney_slots, seeds, id_from_seeds, win_dict

##########SIMULATE##########
def one_tournament(slots, seeds, win_dict, id_from_seeds):
    """
    Simulates one tournament round.

    Parameters:
    - slots (DataFrame): Current Tournament slot data.
    - seeds (dict): Dictionary containing tournament seeds.
    - id_from_seeds (dict): Dictionary mapping seeds to IDs.
    - win_dict (dict): Dictionary containing win probabilities for each

    Returns:
    - winners (list): List of winners in the simulated round.
    - slt (list): List of associated tournament slots.
    """
    winners = []
    slt = []
    for slot, favorite, underdog in zip(slots.Slot, slots.StrongSeed, slots.WeakSeed):
            team_1, team_2 = seeds[favorite], seeds[underdog]
            prob_b = win_dict[(team_1, team_2)]
            winner = np.random.choice([team_1, team_2], p=[1 - prob_b, prob_b])
            winners.append(winner)
            slt.append(slot)
            seeds[slot] = winner   
    return [id_from_seeds[w] for w in winners], slt   

def run_sim(slots, seeds, win_dict, id_from_seeds, brackets = 100000):
    winners = []
    bracket = []
    slts = []
    for sim in range(1, brackets + 1):
            win, slt = one_tournament(slots, seeds, win_dict, id_from_seeds)
            winners.extend(win)
            bracket.extend([sim] * len(win))
            slts.extend(slt)
    result_df = pd.DataFrame({'Bracket': bracket, 'Slot': slts, 'Team': winners})
    
    return result_df

def likeli(teams, tourney_seeds, gender, res):
    r = teams.set_index('TeamID').to_dict()['TeamName']
    seeds = tourney_seeds[tourney_seeds['Tournament'] == gender].set_index('Seed')['TeamID'].to_dict()
    likeli = res[res['Slot'] == 'R6CH'].groupby('Team', dropna=False)['Bracket'].count()/100000
    teamName = [r[seeds[l]] for l in list(likeli.index)]
    lik = pd.DataFrame({'TeamName' : teamName, 'likeli' : likeli}).sort_values('likeli', ascending= False)
    return lik

def most_likeli_tournament(slots, seeds, win_dict, id_from_seeds):
        winners = []
        slt = []
        for slot, favorite, underdog in zip(slots.Slot, slots.StrongSeed, slots.WeakSeed):
                team_1, team_2 = seeds[favorite], seeds[underdog]
                prob_b = win_dict[(team_1, team_2)]
                winner = None
                if prob_b > 0.5:
                    winner = team_2
                else:
                    winner = team_1       
                winners.append(winner)
                slt.append(slot)
                seeds[slot] = winner  
        winners = [id_from_seeds[w] for w in winners]
        slt = slt
        result_df = pd.DataFrame({ 'Slot': slt, 'Team': winners})
 
        return result_df 

def pipeline_ml(massey, reg, tourney_slots, tourney_seeds, tourney_res, teams, best_mod, gender = 'M', quad_weight = False):
    tm_reg = gather_dat(massey, reg)
    X = build_ps_data(tm_reg, tourney_res, qw = quad_weight)
    X.fillna(2,inplace=True)
    slots, seeds, id_from_seeds, win_dict = build_prediction_dat(tourney_slots, tourney_seeds, teams, tm_reg,X, best_mod, gender = gender)
    r = most_likeli_tournament(slots, seeds, win_dict, id_from_seeds)
    return r, best_mod

def build_rankings(win_dict):
    res = pd.DataFrame(win_dict).reset_index()    
##########END-TO-END PIPELINE##########
def pipeline(massey, reg, tourney_slots, tourney_seeds, tourney_res, teams, gender = 'M', quad_weight = False):
    """
    Runs simulations for multiple tournament brackets.

    Parameters:
    - slots (DataFrame): DataFrame containing current tournament slot data.
    - seeds (dict): Dictionary containing current tournament seeds.
    - win_dict (dict): Dictionary containing win probabilities.
    - id_from_seeds (dict): Dictionary mapping seeds to IDs.
    - brackets (int): Number of brackets to simulate.

    Returns:
    - result_df (DataFrame): DataFrame containing simulation results in accordance with tourney style.
    """
    tm_reg = gather_dat(massey, reg)
    X = build_ps_data(tm_reg, tourney_res, qw = quad_weight)
    X.fillna(2,inplace=True)
    acc, fit_mod = model_data(X)
    best_mod = fit_mod[np.argmax(acc)]
    slots, seeds, id_from_seeds, win_dict = build_prediction_dat(tourney_slots, tourney_seeds, teams, tm_reg, X, best_mod, gender = gender)
    r = run_sim(slots, seeds, win_dict, id_from_seeds)
    return r, best_mod, win_dict