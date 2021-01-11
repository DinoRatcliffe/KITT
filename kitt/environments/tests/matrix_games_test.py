import pytest
from unittest.mock import Mock

import numpy as np

from kitpy.environments import matching_pennies
from kitpy.parameters import constant

#############
# Fixtrures #
#############
@pytest.fixture
def head_opponent():
    return constant((Mock(return_value=0), []))

@pytest.fixture
def tail_opponent():
    return constant((Mock(return_value=1), []))


##################
# Initial States #
##################
def test_initial_states(head_opponent):
    initial_states, _ = matching_pennies(head_opponent)

    predicted_observation = np.array([0, 0, 0, 0])
    predicted_state = (head_opponent, True)
    for _ in range(100):
        initial_state, initial_observation = next(initial_states)
        assert (initial_observation == predicted_observation).all(), (
               'should always return the inital empty observation') 
        assert (initial_state == predicted_state), (
               'should always return the internal state') 


##############
# Transition #
##############
@pytest.mark.parametrize('action, predicted_observation, predicted_reward', [
    (0, [1, 0, 1, 0], 1),
    (1, [0, 1, 1, 0], 0)
    ])
def test_transition_default_match_pennies_head(head_opponent,
                                               action,
                                               predicted_observation,
                                               predicted_reward):
    initial_states, transition = matching_pennies(head_opponent)
    state, observation = next(initial_states)

    new_state, new_observation, reward, done, _ = transition(state, action)
    head_opponent, [] = next(head_opponent)

    assert (head_opponent.call_args[0][0] == observation).all()
    head_opponent.assert_called_once()
    assert (new_observation == predicted_observation).all(), (
           'should produce the observation with player and opponent actions')
    assert reward == predicted_reward, ('should get reward for matching pennies')
    assert done, ('should always return done')


@pytest.mark.parametrize('action, predicted_observation, predicted_reward', [
    (0, [1, 0, 0, 1], 0),
    (1, [0, 1, 0, 1], 1)
    ])
def test_transition_default_match_pennies_tail(tail_opponent,
                                               action,
                                               predicted_observation,
                                               predicted_reward):
    initial_states, transition = matching_pennies(tail_opponent)
    state, observation = next(initial_states)

    new_state, new_observation, reward, done, _ = transition(state, action)
    tail_opponent, _ = next(tail_opponent)

    assert (tail_opponent.call_args[0][0] == observation).all()
    tail_opponent.assert_called_once()
    assert (new_observation == predicted_observation).all(), (
           'should produce the observation with player and opponent actions')
    assert reward == predicted_reward, ('should get reward for matching pennies')
    assert done, ('should always return done')


@pytest.mark.parametrize('action, predicted_observation, predicted_reward', [
    (0, [1, 0, 1, 0], 0),
    (1, [0, 1, 1, 0], 1)
    ])
def test_transition_opponent_match_pennies_head(head_opponent,
                                                action,
                                                predicted_observation,
                                                predicted_reward):
    initial_states, transition = matching_pennies(head_opponent, match_pennies=False)
    state, observation = next(initial_states)

    new_state, new_observation, reward, done, _ = transition(state, action)
    head_opponent, _ = next(head_opponent)

    assert (head_opponent.call_args[0][0] == observation).all()
    head_opponent.assert_called_once()
    assert (new_observation == predicted_observation).all(), (
           'should produce the observation with player and opponent actions')
    assert reward == predicted_reward, ('should get reward for matching pennies')
    assert done, ('should always return done')


@pytest.mark.parametrize('action, predicted_observation, predicted_reward', [
    (0, [1, 0, 0, 1], 1),
    (1, [0, 1, 0, 1], 0)
    ])
def test_transition_opponent_match_pennies_tail(tail_opponent,
                                               action,
                                               predicted_observation,
                                               predicted_reward):
    initial_states, transition = matching_pennies(tail_opponent, match_pennies=False)
    state, observation = next(initial_states)

    new_state, new_observation, reward, done, _ = transition(state, action)
    tail_opponent, _ = next(tail_opponent)

    assert (tail_opponent.call_args[0][0] == observation).all()
    tail_opponent.assert_called_once()
    assert (new_observation == predicted_observation).all(), (
           'should produce the observation with player and opponent actions')
    assert reward == predicted_reward, ('should get reward for matching pennies')
    assert done, ('should always return done')


@pytest.mark.parametrize('action', [
    -1,
    2,
    0.02,
    ])
def test_only_valid_actions(head_opponent, action):
    initial_states, transition = matching_pennies(head_opponent)
    state, observation = next(initial_states)

    # should assert that action is either 0 or 1
    with pytest.raises(AssertionError):
        transition(state, action)
