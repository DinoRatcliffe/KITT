"""Collection of function for generating experiance inside environments"""
from typing import Generator, Iterator, Tuple

import pandas as pd

from kitt import take, take_upto
from kitt.types import StateType, ObservationType, TransitionType, ModelPolicyType


################
# Single Agent #
################
def get_experiance(state: StateType,
                   observation: ObservationType,
                   transition: TransitionType,
                   policy: ModelPolicyType, i=-1) -> pd.Series:
    """Get a single experiance

    Given a model policy perform a single action from given state.

    Parameters:
        state: The current internal state of the environment.
        observation: The observation recieved from the current internal state.
        transition: The transition function of the environment.
        policy: The policy of the agent that should act in this environment.

    Returns:
        experiance: A pandas Series that encapsulates the standart Reinforcment
                    learning experiance transition plus stats recieved from the
                    environment and policy.
    """
    action, policy_stats = policy(observation)
    state_prime, observation_prime, reward, terminal, stats = transition(state, action)

    experiance_df = {'state': state,
                     'observation': observation,
                     'action': action,
                     'reward': reward,
                     'state_prime': state_prime,
                     'observation_prime': observation_prime,
                     'terminal': terminal,
                     'policy_stats': policy_stats}

    experiance_df.update(stats)
    experiance_df.update(policy_stats)
    return experiance_df


def experiance_generator(initial_states: Iterator[Tuple[StateType, ObservationType]],
                         transition: TransitionType,
                         policy: ModelPolicyType, i=-1) -> Generator[pd.Series, None, None]:
    """Generator that produces infinite amount of experiance.

    Paramters
        initial_states: A generator that samples from the set of initial states
                        and observations for the environment matching the given
                        transition function.
        transition: The transition function of the environment.
        policy: The policy of the agent that should act in this environment.

    Returns:
        experiance_generator: A Generator the yields pandas Series of each
                              individual experiance
    """
    try:
        initial_state, initial_observation = next(initial_states)
        exp = get_experiance(initial_state, initial_observation, transition, policy)
        yield exp
        while True:
            if exp['terminal']:
                initial_state, initial_observation = next(initial_states)
                exp = get_experiance(initial_state,
                                     initial_observation,
                                     transition,
                                     policy)
                yield exp
            else:
                exp = get_experiance(exp['state_prime'],
                                     exp['observation_prime'],
                                     transition,
                                     policy)
                yield exp
    except StopIteration:
        return


def episodes_generator(initial_state: Iterator[Tuple[StateType, ObservationType]],
                       transition: TransitionType,
                       policy: ModelPolicyType, i=-1) -> Iterator[pd.DataFrame]:
    """Generator that produces infinate episodes

    This generator will generate a whole episode in each iteration, defined as
    generating experiance from an initial state until a terminal state is reached.

    Paramters
        initial_states: A generator that samples from the set of initial states
                        and observations for the environment matching the given
                        transition function.
        transition: The transition function of the environment.
        policy: The policy of the agent that should act in this environment.

    Returns:
        episodes_generator: A Generator the yields a pandas DataFrame of stacked
                            experiance
    """
    while True:
        yield pd.DataFrame(take_upto(lambda experiance: experiance['terminal'],
                                     experiance_generator(initial_state, transition, policy)))


def rollout_generator(initial_state: Iterator[Tuple[StateType, ObservationType]],
                      transition: TransitionType,
                      policy: ModelPolicyType,
                      length: int) -> Iterator[pd.DataFrame]:
    """Generator that produces infinate rollouts

    This generator will generate a rollout of length n in each iteration. A
    rollout can contain terminal states within the rollout and will generate
    new inital states as needed.

    Paramters
        initial_states: A generator that samples from the set of initial states
                        and observations for the environment matching the given
                        transition function.
        transition: The transition function of the environment.
        policy: The policy of the agent that should act in this environment.
        length: The length of the rollout to generate at each iteration

    Returns:
        rollout_generator: A Generator the yields a pandas DataFrame of stacked
                           experiance
    """
    ep_gen = experiance_generator(initial_state,
                                  transition,
                                  policy)
    while True:
        yield pd.DataFrame(take(length, ep_gen))
