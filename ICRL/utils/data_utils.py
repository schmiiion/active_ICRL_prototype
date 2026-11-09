import yaml
import os
import pickle
import numpy as np
import argparse
from copy import deepcopy

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to configs file")
    # parser.add_argument("-t", "--train_flag", help="if training",
    #                     dest="TRAIN_FLAG",
    #                     default='1', required=False)
    parser.add_argument("-d", "--debug_mode", help="whether to use the debug mode",
                        dest="DEBUG_MODE",
                        default=False, required=False)
    parser.add_argument("-p", "--part_data", help="whether to use the partial dataset",
                        dest="PART_DATA",
                        default=False, required=False)
    parser.add_argument("-n", "--num_threads", help="number of threads for loading envs.",
                        dest="NUM_THREADS",
                        default=1, required=False)
    parser.add_argument("-s", "--seed", help="the seed of randomness",
                        dest="SEED",
                        default=123,
                        required=False,
                        type=int)
    parser.add_argument("-l", "--log_file", help="log file", dest="LOG_FILE_PATH", default=None, required=False)
    args = parser.parse_args()
    return args


def load_config(args=None):
    assert os.path.exists(args.config_file), "Invalid configs file {0}".format(args.config_file)
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    return config, args.NUM_THREADS, args.SEED


def load_expert_data(expert_path,
                     num_rollouts=None,
                     add_next_step=True,
                     log_file=None):
    print('Loading expert data from {0}.'.format(expert_path), file=log_file, flush=True)
    file_names = sorted(os.listdir(expert_path))
    # file_names = [i for i in range(29)]
    # sample_names = random.sample(file_names, num_rollouts)
    expert_sum_rewards = []
    expert_obs = []
    expert_acs = []
    expert_rs = []
    num_samples = 0
    if num_rollouts is None or num_rollouts > len(file_names):
        num_rollouts = len(file_names)
    for i in range(num_rollouts):
        # file_name = sample_names[i]
        file_name = file_names[i]
        with open(os.path.join(expert_path, file_name), "rb") as f:
            data = pickle.load(f)
        data_obs = data['original_observations']
        data_acs = data['actions']
        if 'reward' in data.keys():
            data_rs = data['reward']
        else:
            data_rs = None
        if add_next_step:
            total_time_step = data_acs.shape[0] - 1
        else:
            total_time_step = data_acs.shape[0]

        expert_obs_episode = []
        expert_acs_episode = []
        expert_rs_episode = []

        for t in range(total_time_step):
            data_obs_t = data_obs[t]
            data_ac_t = data_acs[t]
            if add_next_step:
                data_obs_next_t = data_obs[t + 1]
                data_ac_next_t = data_acs[t + 1]
            num_samples += 1
            if data_rs is not None:
                data_r_t = data_rs[t]
                if add_next_step:
                    data_r_next_t = data_rs[t + 1]
            else:
                data_r_t = 0
                if add_next_step:
                    data_r_next_t = 0
            if add_next_step:
                data_obs_t_store = [data_obs_t, data_obs_next_t]
                data_acs_t_store = [data_ac_t, data_ac_next_t]
                data_r_t_store = [data_r_t, data_r_next_t]
            else:
                data_obs_t_store = data_obs_t
                data_acs_t_store = data_ac_t
                data_r_t_store = data_r_t
            expert_obs_episode.append(data_obs_t_store)
            expert_acs_episode.append(data_acs_t_store)
            expert_rs_episode.append(data_r_t_store)


        expert_obs.append(np.asarray(expert_obs_episode))
        expert_acs.append(np.asarray(expert_acs_episode))
        expert_rs.append(np.asarray(expert_rs_episode))
        expert_sum_rewards.append(data['reward_sum'])
    expert_avg_sum_reward = np.mean(expert_sum_rewards)
    expert_mean_length = num_samples / len(file_names)
    print('Expert_mean_reward: {0} and Expert_mean_length: {1}.'.format(expert_avg_sum_reward, expert_mean_length),
          file=log_file, flush=True)
    return (expert_obs, expert_acs, expert_rs), expert_avg_sum_reward


def get_input_features_dim(feature_select_names, all_feature_names):
    if feature_select_names is None:
        feature_select_dim = None
    else:
        feature_select_dim = []
        for feature_name in feature_select_names:
            if feature_name == -1:
                feature_select_dim.append(-1)  # -1 indicates don't select
                break
            else:
                feature_select_dim.append(all_feature_names.index(feature_name))
    return feature_select_dim

def sync_envs_normalization(train_env, eval_env) -> None:
    """ Sync observation normalization in eval env to be equally to the train env """
    train_obs_normalizer = train_env.save()["obs_normalizer"]
    eval_obs_normalizer = eval_env.save()["obs_normalizer"]

    eval_obs_normalizer.load_state_dict(
        train_obs_normalizer.state_dict()
    )
    eval_obs_normalizer.eval()


def idx2vector(indices, height, width):
    vector_all = []
    for idx in indices:
        map = np.zeros(shape=[height, width])
        x, y = int(round(idx[0], 0)), int(round(idx[1], 0))
        # if x - idx[0] != 0:
        #     print('debug')
        map[x, y] = 1  # + idx[0] - x + idx[1] - y
        vector_all.append(map.flatten())
    return np.asarray(vector_all)
