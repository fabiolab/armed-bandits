import abc
from abc import ABC
from math import log, sqrt
from random import randint
from typing import Iterator, List

from pydantic import BaseModel


class ActionHistory(BaseModel):
    rewards: int
    played: int
    action_id: int

    @property
    def mean_reward(self) -> float:
        return self.rewards / self.played

    def observe(self, reward: int) -> None:
        self.rewards += reward
        self.played += 1


class BanditSystem(ABC):
    def __init__(self, n_actions: int):
        self.actions: List[ActionHistory] = []
        for i in range(n_actions):
            self.actions.append(ActionHistory(rewards=0, played=1, action_id=i))
        self.actions_played = 1

    def observe(self, action_id: int, reward: int) -> None:
        self.actions[action_id].observe(reward)
        self.actions_played += 1

    @abc.abstractmethod
    def select(self) -> int:
        pass


class RandomBandits(BanditSystem):
    def select(self) -> int:
        return randint(0, len(self.actions))


class UCBBandits(BanditSystem):
    def select(self) -> int:
        return max(
            self.actions,
            key=lambda action: self._get_upper_confidence_bound(action),
        ).action_id

    def _get_upper_confidence_bound(self, action: ActionHistory) -> float:
        return action.rewards / action.played + sqrt(
            2 * log(self.actions_played) / action.played
        )


class SuccessiveEliminationBandits(BanditSystem):
    C = 1           # Inconnu (C>=1 : conservateur, C<1 : agressif )
    DELTA = 0.05    # Probabilité (acceptée) de se tromper (petit : converge plus vite)

    def __init__(self, n_actions: int):
        super().__init__(n_actions)
        self.selection = self._get_selection()

    def select(self) -> int:
        next_action = next(self.selection)
        return next_action

    def _get_selection(self) -> Iterator[int]:
        while True:
            actions_ids = [action.action_id for action in self.actions]
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

        self.actions = [
            action
            for action in self.actions
            if (best_action.mean_reward - action.mean_reward) < 2 * epsilon_t
        ]

        print(f"Il reste {len(self.actions)} actions")
