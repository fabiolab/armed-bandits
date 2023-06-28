import random
from random import randint

from src.bandit_system import BanditSystem

random.seed(18)


class RandomBandits(BanditSystem):
    def select(self) -> int:
        return randint(0, len(self.actions) - 1)
