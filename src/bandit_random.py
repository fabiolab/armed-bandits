from random import randint

from bandit_system import BanditSystem


class RandomBandits(BanditSystem):
    def select(self) -> int:
        return randint(0, len(self.actions) - 1)
