import torch
import gymnasium as gym
from isaaclab.envs import DirectRLEnv
from isaaclab.envs import DirectMARLEnv

class SingleAgentWrapper(DirectRLEnv):
    def __init__(self, env: DirectMARLEnv, active_agent: str, frozen_policy=None):
        # print("--------------- Env Print from SingleAgentWrapper __init__ --------------")
        # print(type(env))
        # print(dir(env))
        self.env: DirectMARLEnv = env

        self.cfg = self.env.cfg
        self.scene = self.env.scene
        self.sim = self.env.sim
        self.render_mode = self.env.render_mode
        self._num_envs = self.env.num_envs
        self._device = self.env.device
        self._is_closed = False

        self.active_agent = active_agent
        self.inactive_agent = [a for a in self.cfg.possible_agents if a != self.active_agent][0]
        self.frozen_policy = frozen_policy

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = self.env.observation_spaces[active_agent]
        self.single_action_space = self.env.action_spaces[active_agent]

        self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space["policy"], self.num_envs
            )
        self.action_space = gym.vector.utils.batch_space(
                self.single_action_space, self.num_envs
            )

    def __getattr__(self, key: str):
        return getattr(self.env, key)
    
    def step(self, actions: torch.Tensor):
        obs = self.env._get_observations()
        if self.frozen_policy:
            with torch.no_grad():
                frozen_actions = self.frozen_policy({"policy": obs[self.inactive_agent]})
        else:
            frozen_actions = torch.zeros(self.num_envs, 2, device=self.device)

        combined_actions = {self.active_agent: actions, self.inactive_agent: frozen_actions}

        obs, rewards, terminated, time_outs, extras = self.env.step(combined_actions)

        obs = {"policy": obs[self.active_agent]}
        rewards = rewards[self.active_agent]

        return obs, rewards, terminated, time_outs, extras

    def reset(self):
        obs, extras = self.env.reset()
        return {"policy": obs[self.active_agent]}, extras

    def _get_observations(self) -> dict:
        obs = self.env._get_observations()
        return {"policy": obs[self.active_agent]}

    def _get_rewards(self) -> torch.Tensor:
        rewards = self.env._get_rewards()
        return rewards[self.active_agent]

    def _get_dones(self) -> tuple:
        dones = self.env._get_dones()
        return dones
    
    def close(self) -> None:
        self._is_closed = True
    
    @property
    def num_envs(self):
        return self._num_envs
    
    @property
    def device(self):
        return self._device
    
    @property
    def unwrapped(self):
        return self