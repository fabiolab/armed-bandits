from random import random
import pandas as pd
import plotly.graph_objects as go

from bandit_random import RandomBandits
from bandit_successive_elimination import SuccessiveEliminationBandits
from bandit_ucb import UCBBandits

NB_ACTIONS = 3
NB_ROUNDS = 500
BEST_ACTION_ID = 0
BEST_ACTION_DELTA = 0.2

bandits = RandomBandits(NB_ACTIONS)
bandits = UCBBandits(NB_ACTIONS)
# bandits = SuccessiveEliminationBandits(NB_ACTIONS)


def play(action_id: int) -> float:
    # Simulate a user interaction with a random reward
    # The best action has an artificial bonus reward
    if action_id == BEST_ACTION_ID:
        return random() + BEST_ACTION_DELTA
    return random()


datas = {i: [] for i in range(NB_ACTIONS)}
datas["rounds"] = []
datas["mean_regret"] = []
datas["instant_regret"] = []

for round in range(NB_ROUNDS):
    action = bandits.select()
    reward = play(action)
    bandits.observe(action, reward)

    datas["rounds"].append(round)

    for action in bandits.actions:
        datas[action.action_id].append(action.mean_reward)

    regret = bandits.actions[BEST_ACTION_ID].mean_reward * (round + 1) - bandits.total_cumulative_reward
    datas["mean_regret"].append(regret / (round + 1))
    datas["instant_regret"].append(bandits.actions[BEST_ACTION_ID].mean_reward - reward)

df = pd.DataFrame(data=datas)


fig = go.Figure()

fig.add_trace(go.Scatter(x=df.rounds, y=df[0], name="Action 0 (Best)"))
fig.add_trace(go.Scatter(x=df.rounds, y=df[1], name="Action 1"))
fig.add_trace(go.Scatter(x=df.rounds, y=df[2], name="Action 2"))
fig.add_trace(go.Scatter(x=df.rounds, y=df["mean_regret"], name="Regret"))
fig.add_trace(go.Scatter(x=df.rounds, y=df["instant_regret"], name="Instant Regret"))
fig.update_traces(mode="lines")

fig.show()

# Questions
# - regret moyen
# - regret instantané
# - regret cumulé
# Successive elimination ne fait pas disparaître de bras
# reward => float

# Ajouter la courbe de la moyenne des récompenses totale (moyenne glissante sur 10 ? 20 ? tours)
