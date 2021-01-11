import pytest
from itertools import repeat, count

import pandas as pd

from kitt.experiance_generators import (
        get_experiance, experiance_generator, episodes_generator,
        rollout_generator)

##################
# Get Experiance #
##################
# So simple not sure if needs more testing?
def test_get_experiance():
    state = 100
    observation = 100
    def transition(state, action): return (state+1,
                                           observation+1,
                                           0,
                                           state>200,
                                           pd.Series([20], index=['score']))
    def policy(state): return (1, {})

    experiance = get_experiance(state, observation, transition, policy)

    assert isinstance(experiance, dict), (
           'should return a dict object of experiance')
    assert experiance['state'] == 100, ('should populate state in experiance')
    assert experiance['observation'] == 100, ('should populate observation in experiance')
    assert experiance['reward'] == 0, ('should populate reward in experiance')
    assert experiance['state_prime'] == 101, (
           'should populate state_prime in experiance')
    assert experiance['observation_prime'] == 101, (
           'should populate observation_prime in experiance')
    assert experiance['action'] == 1, ('should populate experiance with action')
    assert not experiance['terminal'], (
           'should populate experiance with terminal state')
    assert experiance['score'] == 20, (
           'should populate experiance with environment stats') 


########################
# Experiance Generator #
########################
def test_experiance_generator():
    initial_states = repeat((0, 0))
    def transition(state, action): return (state+1,
                                           state+1,
                                           state*2,
                                           state>=10,
                                           pd.Series([20], index=['score']))
    def policy(state): return (1, {})

    experiances = experiance_generator(initial_states, transition, policy)

    # loop thorugh first full experiance -1
    for i in range(10):
        experiance = next(experiances)
        assert experiance['state'] == i, (
               'state should increment by 1 each time through')
        assert experiance['observation'] == i, (
               'observation should increment by 1 each time through')
        assert not experiance['terminal'], (
               'should not return terminal state at this stage')
        assert experiance['reward'] == i*2, ('should give appropriate reward')
        assert experiance['state_prime'] == i+1, (
               'should give appropriate state_prime')
        assert experiance['action'] == 1, ('should give appropriate action')
        assert experiance['score'] == 20, (
               'should populate experiance with environment stats') 

    # should now hit terminal state
    experiance = next(experiances)
    assert experiance['terminal'], ('should hit terminal state')

    experiance = next(experiances)
    assert experiance['state'] == 0, ('should start from new initial_state')
    assert experiance['observation'] == 0, ('should state from new initial_state')


######################
# Episodes Generator #
######################
def test_episodes_generator():
    initial_states = repeat((1, 1))
    def transition(state, action): return (state+1,
                                           state+1,
                                           state*2,
                                           state>=10,
                                           pd.Series([20], index=['score']))
    def policy(state): return (1, {})

    episodes = episodes_generator(initial_states, transition, policy)

    for i in range(3):
        episode = next(episodes)
        assert isinstance(episode, pd.DataFrame), (
                'should return a pandas DataFrame of the experinaces')
        assert len(episode) == 10, (
               'should give a generator of experinances to the terminal state')
        assert episode.terminal.iloc[-1], (
               'should return terminal state at end of episode')


#####################
# Rollout Generator #
#####################
def test_rollout_generator():
    initial_states = repeat((1, 1))
    def transition(state, action): return (state+1,
                                           state+1,
                                           state*2,
                                           state>=10,
                                           pd.Series([20], index=['score']))
    def policy(state): return (1, {})

    rollouts = rollout_generator(initial_states, transition, policy, 13)

    for _ in range(10):
        rollout = next(rollouts)
        assert isinstance(rollout, pd.DataFrame), (
               'should return a pandas dataframe of the rollout')
        assert len(rollout) == 13, ('should generate a rollout of length 13')
