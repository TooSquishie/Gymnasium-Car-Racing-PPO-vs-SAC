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


class ActionNoise(gym.ActionWrapper):
    """
    Adds Gaussian noise to actions to simulate imperfect control.
    """

    def __init__(self, env, sigma=0.05):
        super().__init__(env)
        self.sigma = sigma

    def action(self, action):
        noise = np.random.normal(0, self.sigma, size=action.shape)
        return np.clip(action + noise, -1.0, 1.0)

class ActionDelay(gym.ActionWrapper):
    """
    Adds a small delay to control (executes older actions).
    """
    
    def __init__(self, env, delay=2):
        super().__init__(env)
        self.delay = delay
        self.buffer = deque(
            [np.zeros(env.action_space.shape, dtype=np.float32)] * delay,
            maxlen=delay,
        )
        
    def action(self, action):
        self.buffer.append(action)
        # Use the oldest action in the buffer
        return np.array(self.buffer[0], dtype=np.float32)

class EarlyTerminateOffTrack(gym.Wrapper):
    """
    Ends episode early if car stays in strongly negative reward for too long
    (heuristic for 'off track' or stuck).
    """
    
    def __init__(self, env, patience=30):
        super().__init__(env)
        self.patience = patience
        self.off = 0
        
    def reset(self, **kwargs):
        self.off = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if reward < -0.05:
            self.off += 1
        else:
            self.off = 0
            
        if self.off >= self.patience:
            truncated = True
            
        return obs, reward, terminated, truncated, info

class StuckPenaltyWrapper(gym.Wrapper):
    """
    Penalize the agent for staying in roughly the same visual position
    (e.g., spinning in place doing donuts) for too many steps.
    Works on the processed stacked observation (C, H, W).
    """

    def __init__(
        self,
        env,
        diff_threshold: float = 2.0,
        max_stuck_steps: int = 40,
        penalty_per_step: float = -0.1,
        terminate_when_stuck: bool = True,
    ):
        super().__init__(env)
        self.diff_threshold = diff_threshold
        self.max_stuck_steps = max_stuck_steps
        self.penalty_per_step = penalty_per_step
        self.terminate_when_stuck = terminate_when_stuck

        self.prev_frame = None
        self.stuck_steps = 0

    def reset(self, **kwargs):
        self.prev_frame = None
        self.stuck_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # obs is (C, H, W); just compare the last channel or the mean
        current = obs[-1]  # last frame in the stack

        if self.prev_frame is not None:
            diff = np.mean(np.abs(current.astype(np.float32) - self.prev_frame.astype(np.float32)))

            # If the frame isn't changing much and the reward is not very positive,
            # we consider this "stuck / donut" behaviour.
            if diff < self.diff_threshold and reward <= 0.1:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0
        else:
            diff = None
            self.stuck_steps = 0

        self.prev_frame = current

        # Apply penalty if stuck for too long
        if self.stuck_steps >= self.max_stuck_steps:
            reward += self.penalty_per_step
            if self.terminate_when_stuck:
                truncated = True  # end the episode

        # You can optionally log diagnostic info into 'info'
        info["stuck_steps"] = self.stuck_steps
        return obs, reward, terminated, truncated, info


def make_car_env(seed: int | None = None, render_mode: str | None = None, hard_mode: bool = False):
    """
    Base CarRacing environment + preprocessing wrapper.
    
    render_mode:
      - None: no window (for training / headless eval)
      - "human": show a window (for watching the agent)
      - optional 'hard mode' (noise, delay, early termination)
    """
    env = gym.make("CarRacing-v3", domain_randomize=True, continuous=True, render_mode=render_mode)
    
    if seed is not None:
        env.reset(seed=seed)

    # Wrap with our observation preprocessor
    env = ResizeGrayStack(env, size=(64, 64), stack=4)
    
    if hard_mode:
        env = ActionNoise(env, sigma=0.05)
        env = ActionDelay(env, delay=2)
        env = EarlyTerminateOffTrack(env, patience=30)
    
    env = StuckPenaltyWrapper(env, diff_threshold=2.0, max_stuck_steps=40, penalty_per_step=-0.1, terminate_when_stuck=True)
    
    return env
