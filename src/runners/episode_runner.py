from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

import matplotlib.pyplot as plt
import os
import logging
import shutil
import copy


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.verbose = args.verbose

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, t_episode=0):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        replay_data = []
        if self.verbose:
            if t_episode < 2:
                save_path = os.path.join(self.args.local_results_path,
                                         "pic_replays",
                                         self.args.unique_token,
                                         str(t_episode))
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                os.makedirs(save_path)
                role_color = np.array(['r', 'y', 'b', 'c', 'm', 'g'])
                print(self.mac.role_action_spaces.detach().cpu().numpy())
                logging.getLogger('matplotlib.font_manager').disabled = True
            all_roles = []

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            if self.verbose:
                # These outputs are designed for SMAC
                ally_info, enemy_info = self.env.get_structured_state()
                replay_data.append([ally_info, enemy_info])

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, roles, role_avail_actions = self.mac.select_actions(self.batch, t_ep=self.t,
                                                                         t_env=self.t_env, test_mode=test_mode)
            self.batch.update({"role_avail_actions": role_avail_actions.tolist()}, ts=self.t)

            if self.verbose:
                roles_detach = roles.detach().cpu().squeeze().numpy()
                ally_info = replay_data[-1][0]
                p_roles = np.where(ally_info['health'] > 0, roles_detach,
                                   np.array([-5 for _ in range(self.args.n_agents)]))

                all_roles.append(copy.deepcopy(p_roles))

                if t_episode < 2:
                    figure = plt.figure()

                    print(self.t, p_roles)
                    ally_health = ally_info['health']
                    ally_health_max = ally_info['health_max']
                    if 'shield' in ally_info.keys():
                        ally_health += ally_info['shield']
                        ally_health_max += ally_info['shield_max']
                    ally_health_status = ally_health / ally_health_max
                    plt.scatter(ally_info['x'], ally_info['y'], s=20*ally_health_status, c=role_color[roles_detach])
                    for agent_i in range(self.args.n_agents):
                        plt.text(ally_info['x'][agent_i], ally_info['y'][agent_i], '{:d}'.format(agent_i+1), c='y')

                    enemy_info = replay_data[-1][1]
                    enemy_health = enemy_info['health']
                    enemy_health_max = enemy_info['health_max']
                    if 'shield' in enemy_info.keys():
                        enemy_health += enemy_info['shield']
                        enemy_health_max += enemy_info['shield_max']
                    enemy_health_status = enemy_health / enemy_health_max
                    plt.scatter(enemy_info['x'], enemy_info['y'], s=20*enemy_health_status, c='k')
                    for enemy_i in range(len(enemy_info['x'])):
                        plt.text(enemy_info['x'][enemy_i], enemy_info['y'][enemy_i], '{:d}'.format(enemy_i+1))

                    plt.xlim(0, 32)
                    plt.ylim(0, 32)
                    plt.title('t={:d}'.format(self.t))
                    pic_name = os.path.join(save_path, str(self.t) + '.png')
                    plt.savefig(pic_name)
                    plt.close()

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "roles": roles,
                "role_avail_actions": role_avail_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # if self.verbose:
        #     # These outputs are designed for SMAC
        #     ally_info, enemy_info = self.env.get_structured_state()
        #     replay_data.append([ally_info, enemy_info])

        # Select actions in the last stored state
        actions, roles, role_avail_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions, "roles": roles, "role_avail_actions": role_avail_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if self.verbose:
            return self.batch, np.array(all_roles)

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
