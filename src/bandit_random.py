import random

from src.bandit_system import BanditSystem

random.seed(18)


class RandomBandits(BanditSystem):
    def select(self) -> int:
        return random.choice(self.actions).action_id
