from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bandit_random import RandomBandits
from bandit_successive_elimination import SuccessiveEliminationBandits
from bandit_ucb import UCBBandits

from numpy.random import binomial

NB_ACTIONS = 5
NB_ROUNDS = 50000
BEST_ACTION_ID = 0
BEST_ACTION_DELTA = 0.2
WINDOW_SIZE = 500

bandits = RandomBandits(NB_ACTIONS)
bandits = UCBBandits(NB_ACTIONS)
bandits = SuccessiveEliminationBandits(NB_ACTIONS)


# Generate a list of sublists of a given window size from a source list and compute the average value
def compute_moving_average(data: List[float], window_size: int) -> List[float]:
    return list(np.convolve(data, np.ones(window_size), "valid") / window_size)


def display_data(data: Dict, title: str):
    data["moving_mean_regret"] = compute_moving_average(
        data["instant_regret"], WINDOW_SIZE
    )
    data["moving_mean_regret"] += data["moving_mean_regret"][-WINDOW_SIZE + 1 :]

    df = pd.DataFrame(data=data)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.rounds, y=df["0"], name="Action 0 (Best)"))
    for i in range(1, NB_ACTIONS):
        fig.add_trace(go.Scatter(x=df.rounds, y=df[str(i)], name=f"Action {i}"))
    fig.add_trace(go.Scatter(x=df.rounds, y=df["mean_regret"], name="Regret"))
    fig.add_trace(go.Scatter(x=df.rounds, y=df["mean_reward"], name="Mean Reward"))
    fig.add_trace(
        go.Scatter(x=df.rounds, y=df["ab_mean_reward"], name="AB Mean Reward")
    )
    fig.add_trace(
        go.Scatter(x=df.rounds, y=df["moving_mean_regret"], name="Moving Mean Regret")
    )
    fig.update_traces(mode="lines")
    fig.update_layout(
        title=title, xaxis=dict(title="Rounds"), yaxis=dict(title="Reward")
    )
    fig.show()


def update_viz_data(
    data: Optional[Dict[str, List[float]]], round_number: int
) -> Dict[str, List[float]]:

    if not data:
        data: Dict[str, List[float]] = {str(i): [] for i in range(NB_ACTIONS)}
        data["rounds"] = []
        data["mean_regret"] = []
        data["instant_regret"] = []
        data["mean_reward"] = []
        data["ab_mean_reward"] = []

    data["rounds"].append(round_number)
    for action in bandits.actions:
        data[str(action.action_id)].append(action.mean_reward)
    regret = (
        bandits.actions[BEST_ACTION_ID].mean_reward * (round_number + 1)
        - bandits.total_cumulative_reward
    )
    data["mean_regret"].append(regret / (round_number + 1))
    data["instant_regret"].append(bandits.actions[BEST_ACTION_ID].mean_reward - reward)
    data["mean_reward"].append(bandits.total_cumulative_reward / (round_number + 1))
    data["ab_mean_reward"].append(
        (0.5 + BEST_ACTION_DELTA + (0.5 * (NB_ACTIONS - 1))) / NB_ACTIONS
    )
    return data


def play(action_id: int) -> int:
    # Simulate a user interaction with a random reward
    # The best action has an artificial bonus reward
    if action_id == BEST_ACTION_ID:
        return binomial(1, 0.5 + BEST_ACTION_DELTA)
    return binomial(1, 0.5)


visualization_data = {}
for current_round in range(NB_ROUNDS):
    action_id: int = bandits.select()
    reward: int = play(action_id)
    bandits.observe(action_id, reward)

    visualization_data = update_viz_data(visualization_data, current_round)

display_data(visualization_data, str(bandits))
