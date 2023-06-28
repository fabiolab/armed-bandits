from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.bandit_system import BanditSystem


# Generate a list of sublists of a given window size from a source list and compute the average value
def compute_moving_average(data: List[float], window_size: int) -> List[float]:
    return list(np.convolve(data, np.ones(window_size), "valid") / window_size)


def display_data(data: Dict, title: str, window_size: int, nb_actions: int) -> None:
    data["moving_mean_regret"] = compute_moving_average(
        data["instant_regret"], window_size
    )
    data["moving_mean_regret"] += data["moving_mean_regret"][-window_size + 1 :]

    df = pd.DataFrame(data=data)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.rounds, y=df["0"], name="Action 0 (Best)"))
    for i in range(1, nb_actions):
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
    bandits: BanditSystem,
    reward: int,
    data: Optional[Dict[str, List[float]]],
    round_number: int,
    best_action_id: int,
    best_action_delta: float,
) -> Dict[str, List[float]]:

    nb_actions = len(bandits.actions)

    if not data:
        data: Dict[str, List[float]] = {str(i): [] for i in range(nb_actions)}
        data["rounds"] = []
        data["mean_regret"] = []
        data["instant_regret"] = []
        data["mean_reward"] = []
        data["ab_mean_reward"] = []

    data["rounds"].append(round_number)
    for action in bandits.actions:
        data[str(action.action_id)].append(action.mean_reward)
    regret = (
        bandits.actions[best_action_id].mean_reward * (round_number + 1)
        - bandits.total_cumulative_reward
    )
    data["mean_regret"].append(regret / (round_number + 1))
    data["instant_regret"].append(bandits.actions[best_action_id].mean_reward - reward)
    data["mean_reward"].append(bandits.total_cumulative_reward / (round_number + 1))
    data["ab_mean_reward"].append(
        (0.5 + best_action_delta + (0.5 * (nb_actions - 1))) / nb_actions
    )
    return data
