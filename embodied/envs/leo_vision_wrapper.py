"""
Integration module to connect the LeoRoverEnv with DreamerV3 agent
-----------------------------------------------------------------
This module provides the necessary wrapper to make the Leo Rover environment
compatible with DreamerV3's expectation of observation format.

"""

from embodied.envs.leo_rover_env import LeoRoverEnv
import gym
import numpy as np
import functools
import elements


class FromGymLeoWrapper(gym.Wrapper):
    """Wrapper to adapt a standard gym environment to DreamerV3's expectations."""
    
    def __init__(self, env, obs_key="image", act_key="action"):
        super().__init__(env)
        self._env = env
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None
    
    @functools.cached_property
    def obs_space(self):
        """Convert gym observation_space to DreamerV3's obs_space format."""
        spaces = {self._obs_key: self._convert(self._env.observation_space['image'])}
        return {
            **spaces,
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
    
    @functools.cached_property
    def act_space(self):
        """Convert gym action_space to DreamerV3's act_space format."""
        spaces = {self._act_key: self._convert(self._env.action_space)}
        spaces["reset"] = elements.Space(bool)
        return spaces
    
    def step(self, action):
        """Adapt the step method to DreamerV3's expectations."""
        if action["reset"] or self._done:
            self._done = False
            obs = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)
        
        # Extract the action from the dictionary
        act = action[self._act_key]
        obs, reward, done, info = self._env.step(act)
        self._done = done
        self._info = info
        
        # Convert observation to DreamerV3 format
        return self._obs(
            obs,
            reward,
            is_last=bool(done),
            is_terminal=bool(info.get("is_terminal", done)),
        )
    
    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        """Format the observation according to DreamerV3's expectations."""
        return {
            self._obs_key: obs["image"],
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }
    
    def _convert(self, space):
        """Convert gym space to elements.Space."""
        if isinstance(space, gym.spaces.Discrete):
            return elements.Space(np.int32, (), 0, space.n)
        elif isinstance(space, gym.spaces.Box):
            return elements.Space(space.dtype, space.shape, space.low, space.high)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")


def LeoRover(task, **kwargs):
    """Factory function that creates and wraps a LeoRoverEnv for DreamerV3.
    
    Args:
        task: Task name (required by DreamerV3 but not used directly by LeoRoverEnv)
        **kwargs: Additional arguments to pass to LeoRoverEnv
    
    Returns:
        A wrapped LeoRoverEnv compatible with DreamerV3
    """
    # We don't use the task parameter directly as LeoRoverEnv doesn't need it
    env = LeoRoverEnv(**kwargs)
    # Add the DreamerV3 wrapper that converts the gym API to DreamerV3's expectations
    env = FromGymLeoWrapper(env)
    return env
