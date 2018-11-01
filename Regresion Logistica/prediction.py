import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
folder = 'data'

# calcular el rating ELO de cada equipo 
def calc_elo(win_team, lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff  * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    #modificar el valor k dependiendo del rango 
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank

# Iniciar los datos 
def initialize_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')

    print (team_stats1.info())
    return team_stats1.set_index('Team', inplace=False, drop=True)

def get_elo(team):
    try:
        return team_elos[team]
    except:
        #Si no hay un ELO ponerle el de base 1600 
        team_elos[team] = base_elo
        return team_elos[team]

def  build_dataSet(all_data):
    print("Building data set..")
    for index, row in all_data.iterrows():

        Wteam = row['WTeam']
        Lteam = row['LTeam']

        #Agarrar el primer rating ELO de cada equipo  
        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)

        # Se le sube 100 puntos en el ELO al equipo que es local 
        if row['WLoc'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100

        # Se pone el rating ELO como una cualidad de cada equipo
        team1_features = [team1_elo]
        team2_features = [team2_elo]
        
        # se agregan los datos de basketballreference.com
        for key, value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)
        for key, value in team_stats.loc[Lteam].iteritems():
            team2_features.append(value)

        # Distribuir al azar las caracteristicas de los equipos
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)

        # Se actualiza el rating ELO
        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank

    return np.nan_to_num(X), np.array(y)

def predict_winner(team_1, team_2, model):
    features = []

    # equipo 1 = visitante
    features.append(get_elo(team_1))
    for key, value in team_stats.loc[team_1].iteritems():
        features.append(value)

    # equipo 2 = local
    features.append(get_elo(team_2) + 100)
    for key, value in team_stats.loc[team_2].iteritems():
        features.append(value)

    features = np.nan_to_num(features)
    return model.predict_proba([features])

if __name__ == '__main__':
    
    Mstat = pd.read_csv(folder + '/MiscellaneousStats.csv')
    Ostat = pd.read_csv(folder + '/OpponentPerGameStats.csv')
    Tstat = pd.read_csv(folder + '/TeamPerGameStats.csv')

    team_stats = initialize_data(Mstat, Ostat, Tstat)

    result_data = pd.read_csv(folder + '/17-18RegularResult.csv')
    X, y = build_dataSet(result_data)

    # hacer el modelo
    print("Fitting on %d game samples.." % len(X))

    model = LogisticRegression()
    model.fit(X, y)

    # probar el accuracy usando cross-validation 
    print("Doing cross-validation..")
    print(cross_val_score(model, X, y, cv = 10, scoring='accuracy', n_jobs=-1).mean())


    #Predice los playoffs 17-18
    print('Predicting on new schedule..')
    schedule1617 = pd.read_csv(folder + '/17-18PlayoffsSchedule.csv')
    result = []
    for index, row in schedule1617.iterrows():
        team1 = row['Vteam']
        team2 = row['Hteam']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            result.append([winner, loser, prob])
        else:
            winner = team2
            loser = team1
            result.append([winner, loser, 1 - prob])

    with open('17-18PlayoffsResult.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['win', 'lose', 'probability'])
        writer.writerows(result)










