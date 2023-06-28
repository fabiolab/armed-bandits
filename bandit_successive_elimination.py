from math import log, sqrt
from typing import Iterator, List

from bandit_system import BanditSystem


class SuccessiveEliminationBandits(BanditSystem):
    C = 1           # Unkonwn at start (C>=1 : conservative, C<1 : agressive )
    DELTA = 0.05    # Accepted error probability (the less it is, the faster it converges)

    def __init__(self, n_actions: int):
        super().__init__(n_actions)
        self.selection = self._get_selection()
        self.eliminated_action_ids: List[int] = []

    def select(self) -> int:
        next_action = next(self.selection)
        return next_action

    def _get_selection(self) -> Iterator[int]:
        while True:
            actions_ids = [
                action.action_id
                for action in self.actions
                if action.action_id not in self.eliminated_action_ids
            ]
            for action_id in actions_ids:
                yield action_id

            self._check_and_eliminate()

    def _check_and_eliminate(self) -> None:
        best_action = max(self.actions, key=lambda act: act.mean_reward)

        # Confidence interval
        epsilon_t = sqrt(
            log(self.C * len(self.actions) * best_action.played**2 / self.DELTA)
            / best_action.played
        )

        self.eliminated_action_ids = [
            action.action_id
            for action in self.actions
            if (best_action.mean_reward - action.mean_reward) >= 2 * epsilon_t
        ]
