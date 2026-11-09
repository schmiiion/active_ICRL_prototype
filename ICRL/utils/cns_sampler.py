import numpy as np
import torch

class ConstrainedRLSampler:
    """
    Sampling based on the policy
    """

    def __init__(self, rollouts, env, env_id):
        self.rollouts = rollouts
        self.env = env
        self.actor = None
        self.env_id = env_id

    def sample_from_agent(self, actor, new_env):
        self.env = new_env
        self.actor = actor

        all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
        sum_rewards, lengths = [], []
        for i in range(self.rollouts):
            # Avoid double reset, as VecEnv are reset automatically
            if i == 0:
                obs, info = self.env.reset()

            done, state = False, None
            episode_sum_reward = 0.0
            episode_length = 0

            origin_obs_episode = []
            obs_episode = []
            acs_episode = []
            rs_episode = []
            while not done:
                with torch.no_grad():
                    action = self.actor.predict(obs, deterministic=False)
                origin_obs_episode.append(info.get("original_obs", obs))
                obs_episode.append(obs[0].detach().cpu().numpy().copy())
                acs_episode.append(action[0].detach().cpu().numpy().copy())
                obs, reward, cost, terminated, truncated, info = self.env.step(action)
                done = bool(terminated.item() or truncated.item())
                rs_episode.append(reward)

                episode_sum_reward += reward
                episode_length += 1

            origin_obs_episode = np.squeeze(np.array(origin_obs_episode), axis=1)
            obs_episode = np.squeeze(np.array(obs_episode), axis=1)
            acs_episode = np.squeeze(np.array(acs_episode), axis=1)
            rs_episode = np.squeeze(np.asarray(rs_episode))
            all_orig_obs.append(origin_obs_episode)
            all_obs.append(obs_episode)
            all_acs.append(acs_episode)
            all_rs.append(rs_episode)

            sum_rewards.append(episode_sum_reward)
            lengths.append(episode_length)

        return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, lengths
