import os
import gymnasium as gym
import numpy as np
import torch

class ASTEnv(gym.Env):
    def __init__(self, env, policy_fn, px, risk_fn, risk_target, horizon, xscale=1, reset_kwargs={"seed":0}, save=False, save_dir=None, noise="actions",  noise_idxs=None, event_reward=0.0, save_every=1000, action_space=None, observation_space=None):
        # self.policy = policy
        self.save_every = save_every
        self.policy_fn = policy_fn
        self.env = env
        self.px = px
        self.observation_space = self.env.observation_space
        self.risk_fn = risk_fn
        self.risk_target = risk_target
        self.reset_kwargs = reset_kwargs
        self.noise = noise
        self.horizon = horizon
        self.event_reward = event_reward
        self.xscale = xscale
        
        self.obs = None
        obs, info = self.env.reset()
        self.obs = obs
        act = self.env.action_space.sample()

        if noise_idxs is None and self.noise == 'actions':
            self.noise_idxs = np.arange(len(act))
        elif noise_idxs is None and self.noise == 'observations':
            self.noise_idxs = np.arange(len(obs))
        else:
            self.noise_idxs = noise_idxs

        self.action_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(len(self.noise_idxs),), dtype=np.float32)
        
        self.save = save
        self.tstep = 0
        self.batch_size = 0
        self.save_dir = save_dir
        self.action_trajectory = torch.zeros((horizon, *self.env.action_space.shape))
        self.observation_trajectory = torch.zeros((horizon, *self.env.observation_space.shape))
        self.action_data = []
        self.observation_data = []
        self.eval = False

    def reset(self, **kwargs):
        # Save previous trajectories
        if self.save:
            if self.save_dir is not None and self.tstep > 0:
                self.action_data.append(self.action_trajectory[:self.tstep])
                self.observation_data.append(self.observation_trajectory[:self.tstep])
                self.batch_size += 1
        
                if self.batch_size == self.save_every:
                    action_path = os.path.join(self.save_dir, 'disturbances.pt')
                    observation_path = os.path.join(self.save_dir, 'observations.pt')
                    torch.save(self.action_data, action_path)
                    torch.save(self.observation_data, observation_path)
                    self.action_data = []
                    self.observation_data = []
                    self.batch_size = 0

        
        obs, info = self.env.unwrapped.reset(options=self.reset_kwargs)
        self.obs = obs

        # Reset trajectories
        self.action_trajectory.zero_()
        self.observation_trajectory.zero_()
        self.tstep = 0
        return self.obs, info
            

    def step(self, action):
        if self.save:
            self.action_trajectory[self.tstep] = torch.tensor(action)
            self.observation_trajectory[self.tstep] = torch.tensor(self.obs)

        o = self.obs

        if self.noise == "actions":
            a = self.policy_fn(o)
            a[self.noise_idxs] += self.xscale*action

        else:
            o[self.noise_idxs] += self.xscale*action
            a = self.policy_fn(o)

        obs, r_env, terminated, truncated, info = self.env.step(a)
        # print(self.tstep)
        # print(terminated)
        # print(truncated)
        self.obs = obs

        self.tstep += 1

        is_failure = self.risk_fn(obs) >= self.risk_target


        terminal = is_failure or self.tstep >= self.horizon

        if terminal and is_failure:
            r = self.event_reward
        elif terminal:
            r = -10*(self.risk_target - self.risk_fn(obs))
            print(f"risk: {self.risk_fn(obs)}")
        else:
            #print(action)
            r = self.px.log_prob(torch.tensor(action)).sum()

        if self.eval:
            terminal = False

        return obs, r, terminal, terminal, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __getattr__(self, attr):
        return getattr(self.env, attr)
    
    