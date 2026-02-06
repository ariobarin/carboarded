from dataclasses import dataclass

import numpy as np


@dataclass
class RuleBasedParams:
    steer_gain: float = 1.5
    throttle_base: float = 0.2
    throttle_gain: float = 0.7
    brake_dist: float = 0.4
    throttle_min: float = 0.05


class RuleBasedAgent:
    def __init__(self, params: RuleBasedParams):
        self.params = params

    def act(self, obs: np.ndarray) -> np.ndarray:
        left = float(obs[0] + obs[1]) / 2.0
        right = float(obs[3] + obs[4]) / 2.0
        front = float(obs[2])

        steer = (right - left) * self.params.steer_gain
        steer = float(np.clip(steer, -1.0, 1.0))

        throttle = self.params.throttle_base + self.params.throttle_gain * front
        if front < self.params.brake_dist:
            throttle *= front / max(self.params.brake_dist, 1e-6)
        throttle = float(np.clip(throttle, self.params.throttle_min, 1.0))

        return np.array([steer, throttle], dtype=np.float32)
