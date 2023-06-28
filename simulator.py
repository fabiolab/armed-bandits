from src.bandit_random import RandomBandits
from src.bandit_successive_elimination import SuccessiveEliminationBandits
from src.bandit_ucb import UCBBandits
from src.utils import display_data, update_viz_data
from numpy.random import binomial


NB_ACTIONS = 5
NB_ROUNDS = 50000
BEST_ACTION_ID = 0
BEST_ACTION_DELTA = 0.2
WINDOW_SIZE = 500

bandits = RandomBandits(NB_ACTIONS)
bandits = UCBBandits(NB_ACTIONS)
bandits = SuccessiveEliminationBandits(NB_ACTIONS)


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

    visualization_data = update_viz_data(
        bandits=bandits,
        reward=reward,
        data=visualization_data,
        round_number=current_round,
        best_action_id=BEST_ACTION_ID,
        best_action_delta=BEST_ACTION_DELTA,
    )

display_data(
    data=visualization_data,
    title=str(bandits),
    window_size=WINDOW_SIZE,
    nb_actions=NB_ACTIONS,
)
