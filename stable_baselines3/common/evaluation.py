import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv

found_rl_agent = True
try:
    from rl_agent.envs.pettingzoo_env import FlatlandPettingZooEnv
except ModuleNotFoundError:
    warnings.warn(
        "Couldn't import 'rl_agent.envs.pettingzoo_env'! Thus, excluding method 'evaluate_marl_policy().'"
    )
    found_rl_agent = False


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.monitor import Monitor

    if isinstance(env, VecEnv):
        assert (
            env.num_envs == 1
        ), "You must pass only one environment when using this function"
        is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]
    else:
        is_monitor_wrapped = is_wrapped(env, Monitor)

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    episode_rewards, episode_lengths = [], []
    not_reseted = True
    while len(episode_rewards) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(env, VecEnv) or not_reseted:
            obs = env.reset()
            not_reseted = False
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(
                obs, state=state, deterministic=deterministic
            )
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()

        if is_monitor_wrapped:
            # Do not trust "done" with episode endings.
            # Remove vecenv stacking (if any)
            if isinstance(env, VecEnv):
                info = info[0]
            if "episode" in info.keys():
                # Monitor wrapper includes "episode" key in info if environment
                # has been wrapped with it. Use those rewards instead.
                episode_rewards.append(info["episode"]["r"])
                episode_lengths.append(info["episode"]["l"])
        else:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


if found_rl_agent:
    import gym.vector

    def evaluate_marl_policy(
        model: "base_class.BaseAlgorithm",
        env: FlatlandPettingZooEnv,
        num_robots: int,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], None]
        ] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        """
        Runs policy for ``n_eval_episodes`` episodes and returns average reward.
        This is made to work only with one env.

        .. note::
            If environment has not been wrapped with ``Monitor`` wrapper, reward and
            episode lengths are counted as it appears with ``env.step`` calls. If
            the environment contains wrappers that modify rewards or episode lengths
            (e.g. reward scaling, early episode reset), these will affect the evaluation
            results as well. You can avoid this by wrapping environment with ``Monitor``
            wrapper before anything else.

        :param model: The RL agent you want to evaluate.
        :param env: The gym environment. In the case of a ``VecEnv``
            this must contain only one environment.
        :param num_robots: Number of robots in the environment
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param deterministic: Whether to use deterministic or stochastic actions
        :param render: Whether to render the environment or not
        :param callback: callback function to do additional checks,
            called after each step. Gets locals() and globals() passed as parameters.
        :param reward_threshold: Minimum expected reward per episode,
            this will raise an error if the performance is not met
        :param return_episode_rewards: If True, a list of rewards and episode lengths
            per episode will be returned instead of the mean.
        :param warn: If True (default), warns user about lack of a Monitor wrapper in the
            evaluation environment.
        :return: Mean reward per episode, std of reward per episode.
            Returns ([float], [int]) when ``return_episode_rewards`` is True, first
            list containing per-episode rewards and second containing per-episode lengths
            (in number of steps).
        """
        # is_monitor_wrapped = False
        # # Avoid circular import
        # from stable_baselines3.common.env_util import is_wrapped
        # from stable_baselines3.common.monitor import Monitor

        # if isinstance(env, VecEnv):
        #     assert (
        #         env.num_envs == 1
        #     ), "You must pass only one environment when using this function"
        # is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]
        # else:
        #     is_monitor_wrapped = is_wrapped(env, Monitor)

        # if not is_monitor_wrapped and warn:
        #     warnings.warn(
        #         "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
        #         "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
        #         "Consider wrapping environment first with ``Monitor`` wrapper.",
        #         UserWarning,
        #     )

        is_vec_env = isinstance(env, (VecEnv, gym.vector.VectorEnv))

        episode_rewards, episode_lengths = [], []
        not_reseted = True
        agents = env.agents if not is_vec_env else get_vecagent_list(env)

        while len(episode_rewards) < n_eval_episodes:
            # Number of loops here might differ from true episodes
            # played, if underlying wrappers modify episode lengths.
            # Avoid double reset, as VecEnv are reset automatically.
            if not_reseted:
                obs = env.reset()
                not_reseted = False

            done_count, state = (
                0,
                None,
            )  # state later on necessary for recurrent NN
            episode_reward = 0.0
            episode_length = 0
            success_count = 0
            tmp_episode_length = (
                []
            )  # temporarily save steps when a robot is done in order to calc mean

            done_agents = []

            while done_count != num_robots:
                if is_vec_env and type(obs) == np.ndarray:
                    obs = vecenv_listarray2dict(agents, obs)

                actions = {
                    agent: model.predict(
                        obs[agent], state=state, deterministic=deterministic
                    )[0]
                    for agent in agents
                }
                if is_vec_env:
                    actions = vecenv_action_dict2array(actions, done_agents)

                obs, rewards, dones, infos = env.step(actions)
                if is_vec_env:
                    (
                        obs,
                        rewards,
                        dones,
                        infos,
                    ) = vecenv_multiple_listsarrays2dicts(
                        agents, [obs, rewards, dones, infos]
                    )

                curr_dones, curr_done_agents = extract_dones_2(
                    infos, done_agents
                )
                done_count += curr_dones  # counts the number of finished robots

                episode_reward += sum_rewards(rewards, done_agents)

                # update counters if a robot finishes
                if curr_dones > 0:
                    for agent in curr_done_agents:
                        done_agents.append(agent)
                    tmp_episode_length += [episode_length] * curr_dones
                    curr_successes = extract_successes(infos)
                    assert curr_successes <= curr_dones
                    success_count += curr_successes

                episode_length += 1

                if callback is not None:
                    callback(locals(), globals())

                if render:
                    env.render()

            # if is_monitor_wrapped:
            #     # Do not trust "done" with episode endings.
            #     # Remove vecenv stacking (if any)
            #     if isinstance(env, VecEnv):
            #         info = info[0]
            #     if "episode" in info.keys():
            #         # Monitor wrapper includes "episode" key in info if environment
            #         # has been wrapped with it. Use those rewards instead.
            #         episode_rewards.append(info["episode"]["r"])
            #         episode_lengths.append(info["episode"]["l"])
            # else:
            episode_rewards.append(episode_reward / num_robots)
            episode_lengths.append(sum(tmp_episode_length) / num_robots)

            if not is_vec_env:
                obs = env.reset()

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, (
                "Mean reward below threshold: "
                f"{mean_reward:.2f} < {reward_threshold:.2f}"
            )
        if return_episode_rewards:
            return episode_rewards, episode_lengths
        return mean_reward, std_reward


def get_vecagent_list(env):
    from stable_baselines3.common.vec_env import VecNormalize
    from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv

    if isinstance(env, VecNormalize):
        return env.venv.par_env.agents
    elif isinstance(env, MarkovVectorEnv):
        return env.par_env.possible_agents
    else:
        raise TypeError("Env Type not known")


def vecenv_listarray2dict(agent_list: list, data: np.ndarray) -> dict:
    assert type(data) is np.ndarray or type(data) is list
    return {agent: data[i] for i, agent in enumerate(agent_list)}


def vecenv_action_dict2array(data: dict, done_agents: list) -> np.ndarray:
    assert type(data) is dict
    return np.stack(
        [
            data_array if agent not in done_agents else np.array([0.0, 0.0])
            for agent, data_array in data.items()
        ]
    )


def vecenv_multiple_listsarrays2dicts(agent_list: list, data: list) -> tuple:
    return tuple(
        vecenv_listarray2dict(agent_list, data_type) for data_type in data
    )


def extract_dones(dones: Dict[str, bool]) -> Tuple[int, List[str]]:
    """Currently not used as SS SB3 VecEnv wrapper doesn't return dones correctly"""
    return sum(dones.values()), [agent for agent, done in dones.items() if done]


def extract_dones_2(
    infos: Dict[str, Any], done_agents: List[str]
) -> Tuple[int, List[str]]:
    done_agents = [
        agent
        for agent, info in infos.items()
        if "done_reason" in info and agent not in done_agents
    ]
    return len(done_agents), done_agents


def extract_successes(infos: Dict[str, Any]) -> int:
    return sum(
        reason["done_reason"] == 2
        for reason in infos.values()
        if "done_reason" in reason
    )


def sum_rewards(rewards: Dict[str, float], done_agents: List[str]) -> float:
    return sum(
        reward for agent, reward in rewards.items() if agent not in done_agents
    )
