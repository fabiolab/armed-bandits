from math import log, sqrt

from bandit_system import ActionHistory, BanditSystem


class UCBBandits(BanditSystem):
    def select(self) -> int:
        return max(
            self.actions,
            key=lambda action: self._get_upper_confidence_bound(action),
        ).action_id

    def _get_upper_confidence_bound(self, action: ActionHistory) -> float:
        return action.cumulative_reward / action.played + sqrt(
            2 * log(self.actions_played) / action.played
        )
