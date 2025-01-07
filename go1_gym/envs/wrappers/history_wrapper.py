import gym
import torch


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.obs_history_length = self.env.cfg.env.num_observation_history
        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(
            self.env.num_envs,
            self.num_obs_history,
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )
        self.num_privileged_obs = self.num_privileged_obs

    def step(self, action):
        # privileged information and observation history are stored in info
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]
        feasibility_obs = info["feasibility_obs"]
        feasibility_targets = info["feasibility_targets"]
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)

        obs_dict = {
            "obs": obs,
            "privileged_obs": privileged_obs,
            "obs_history": self.obs_history,
            "feasibility_obs": feasibility_obs,
            "feasibility_targets": feasibility_targets,
        }

        return obs_dict, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        feasibility_obs = self.env.get_feasibility_observations()
        feasibility_targets = self.env.get_feasibility_targets()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)

        return {
            "obs": obs,
            "privileged_obs": privileged_obs,
            "obs_history": self.obs_history,
            "feasibility_obs": feasibility_obs,
            "feasibility_targets": feasibility_targets,
        }

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        feasibility_obs = self.env.get_feasibility_observations()
        feasibility_targets = self.env.get_feasibility_targets()
        self.obs_history[:, :] = 0

        return {
            "obs": ret,
            "privileged_obs": privileged_obs,
            "obs_history": self.obs_history,
            "feasibility_obs": feasibility_obs,
            "feasibility_targets": feasibility_targets,
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import trange

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.mini_cheetah.mini_cheetah_config import \
        config_mini_cheetah
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    config_mini_cheetah(Cfg)
    test_env = gym.make("VelocityTrackingEasyEnv-v0", cfg=Cfg)
    env = HistoryWrapper(test_env)

    env.reset()
    action = torch.zeros(test_env.num_envs, 12)
    for i in trange(3):
        obs, rew, done, info = env.step(action)
        print(obs.keys())
        print(f"obs: {obs['obs']}")
        print(f"privileged obs: {obs['privileged_obs']}")
        print(f"obs_history: {obs['obs_history']}")
        print(f"feasibility obs: {obs['feasibility_obs']}")
        print(f"feasibility targets: {obs['feasibility_targets']}")

        img = env.render('rgb_array')
        plt.imshow(img)
        plt.show()
