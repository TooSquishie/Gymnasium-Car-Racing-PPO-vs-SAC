import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class ResizeGrayStack(gym.ObservationWrapper):
    """
    Converts CarRacing RGB frames (H, W, 3) into stacked grayscale frames (C, H, W),
    where C = stack size, e.g. 4.
    """
    def __init__(self, env, size=(64, 64), stack=4):
        super().__init__(env)
        self.size = size
        self.stack = stack
        self.frames = deque(maxlen=stack)

        # Channels-first observation space (C, H, W), uint8
        c = stack
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c, size[0], size[1]),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        """
        Gymnasium-style reset: returns (obs, info).
        We call the base env reset, then pass the obs through our observation() fn.
        """
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    def observation(self, obs):
        """
        This is REQUIRED for ObservationWrapper.
        It will be called automatically after reset() and step().
        obs is (H, W, 3) RGB from CarRacing; we return (C, H, W) uint8.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize
        gray = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)

        # Initialize buffer on first call
        if len(self.frames) == 0:
            for _ in range(self.stack):
                self.frames.append(gray)
        else:
            self.frames.append(gray)

        stacked = np.stack(self.frames, axis=0)  # (C, H, W)
        return stacked


def make_car_env(seed: int | None = None, render_mode: str | None = None):
    """
    Base CarRacing environment + preprocessing wrapper.
    
    render_mode:
      - None: no window (for training / headless eval)
      - "human": show a window (for watching the agent)
    """
    env = gym.make("CarRacing-v3", domain_randomize=True, continuous=True, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)

    # Wrap with our observation preprocessor
    env = ResizeGrayStack(env, size=(64, 64), stack=4)
    return env
