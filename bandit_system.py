import abc
from abc import ABC
from typing import List

from pydantic import BaseModel


class ActionHistory(BaseModel):
    cumulative_reward: int
    played: int
    action_id: int

    @property
    def mean_reward(self) -> float:
        return self.cumulative_reward / self.played

    def observe(self, reward: float) -> None:
        self.cumulative_reward += reward
        self.played += 1


class BanditSystem(ABC):
    def __init__(self, n_actions: int):
        self.actions: List[ActionHistory] = []
        for i in range(n_actions):
            self.actions.append(
                ActionHistory(cumulative_reward=0, played=1, action_id=i)
            )
        self.actions_played: int = 1
        self.total_cumulative_reward: float = 0

    def observe(self, action_id: int, reward: float) -> None:
        self.actions[action_id].observe(reward)
        self.actions_played += 1
        self.total_cumulative_reward += reward

    @abc.abstractmethod
    def select(self) -> int:
        pass
