from random import randint
import pandas as pd
from bandits import SuccessiveEliminationBandits, UCBBandits
import plotly.express as px

NB_ACTIONS = 3
NB_ROUNDS = 100

bandits = UCBBandits(NB_ACTIONS)
bandits = SuccessiveEliminationBandits(NB_ACTIONS)


def play(action_id: int) -> int:
    return randint(0, 1)


# Faire bcp de simulations
# Tracer la moyenne de la récompense du bras joué à chaque tour
#   => conserver la trace de la récompense globale
#   => il s'agit de tracer le regret = l'itération courante * la moyenne reward meilleur bras - la somme des rewards moyens (depuis le début)

# La moyenne de la somme des récompense jusqu'à ce tour du meilleur bras - la moyenne des récompenses des bras joués pour de vrai jusqu'à ce tour
# = Regrets cumulé

# Regret instantané : moyenne du meilleur - moyenne du bras joué à ce tour

datas = {i: [] for i in range(NB_ACTIONS)}
datas["rounds"] = []
for round in range(NB_ROUNDS):
    action = bandits.select()
    rewards = play(action)
    bandits.observe(action, rewards)
    datas["rounds"].append(round)
    for action in bandits.actions:
        datas[action.action_id].append(action.mean_reward)


df = pd.DataFrame(data=datas)

fig = px.line(x=df.rounds, y=[df[0], df[1], df[2]], title="Mean rewards")
fig.show()
