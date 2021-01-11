"""Set of game theory matrix games."""
from typing import Tuple

import numpy as np
import pandas as pd

from kitt.types import (TransitionType,
                         ModelPolicyDiscreetGenerator,
                         InitialStateGeneratorType)

def mp_transition(state: Tuple[ModelPolicyDiscreetGenerator, bool],
                  action: int) -> Tuple[Tuple[ModelPolicyDiscreetGenerator, bool],
                                        np.ndarray,
                                        float,
                                        bool,
                                        pd.Series]:
    """Transition function for the matrix game Matching Pennies

    Parameters:
        state: containing the opponents policy generator and if the player
               is currently trying to match pennies.
        action: 0 play head, 1 play tails.
    """

    assert action in [0, 1], ('environment only accepts actions 0 or 1')

    # initialise state
    opponent_policy_generator = state[0]
    match_pennies = state[1]

    observation = np.array([0, 0, 0, 0])

    opponent_policy, _ = next(opponent_policy_generator)
    opponent_action = opponent_policy(observation)

    pennies_match = opponent_action == action

    if match_pennies:
        reward = 1 if pennies_match else 0
    else:
        reward = 1 if not pennies_match else 0

    score = reward

    new_observation = [0, 0, 0, 0]
    new_observation[action] = 1
    new_observation[opponent_action + 2] = 1

    step_data = pd.Series([score], index=['score'])

    return (state,
            np.array(new_observation),
            reward,
            True,
            step_data)


def mp_initial_state_sequence(opponent_policy: ModelPolicyDiscreetGenerator,
                              match_pennies: bool) -> InitialStateGeneratorType:
    """Initial state, observation generator.

    Generator that sample from the set of initial game states, also
    returns the observation for the initial state.
    """
    while True:
        yield (opponent_policy, match_pennies), np.array([0, 0, 0, 0])


def matching_pennies(opponent_policy: ModelPolicyDiscreetGenerator,
                     match_pennies: bool = True) \
        -> Tuple[InitialStateGeneratorType, TransitionType]:
    """Helper function that create the matching_pennies env

    Returns:
        inital_state_sequence: A generator that gives random initial state.
        transition: A transition function that defines the matching_pennies dynamics.
    """
    return (mp_initial_state_sequence(opponent_policy, match_pennies),
            mp_transition)
