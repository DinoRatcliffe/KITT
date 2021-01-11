import pytest
from collections import defaultdict

from numpy.random import dirichlet, randint

from kitt.policies import (greedy_policy, stochastic_policy, 
                           uniform_random_policy, epsilon_greedy)

probability_fudge_factor = 0.01

##########
# Greedy #
##########
@pytest.mark.parametrize('values, max_action', [
    ([100, 200], 1),
    ([200, 100], 0),
    ([2200, 11000], 1),
    ([0.01, 0.01, 0.01, 0.01, 0.01, 0.02], 5),
    ])
def test_greedy(values, max_action):
    action = greedy_policy(values)
    assert action[0] == max_action, ('should select highest value action')


"""
Test probability here by doing many trials, I don't like the idea of mocking 
random when the implementation may change in the future, this test will work 
whatever method is used to create the psudo random numbers
"""
@pytest.mark.parametrize('values, ties_idx', [
    ([100, 100], [0, 1]),
    ([100, 1000, 93, 10, 1000], [1, 4]),
    ([0.01, 0.00, 0.01, 0.01, 0.01, 0.01], [0, 2, 3, 4, 5]),
    ])
def test_greedy_random_tie_break(values, ties_idx):
    counts = defaultdict(int)
    trials = 20000
    for _ in range(trials):
        action = greedy_policy(values)[0]
        counts[action] += 1

    prob = 1/len(ties_idx)
    for action, count in counts.items():
        assert action in ties_idx, ('should return a max action')
        assert abs(count / trials - prob) < probability_fudge_factor, (
               'should be around expected probability')


##############
# Stochastic #
##############
# just sample some distributions each time through tests
@pytest.mark.parametrize('pi', [dirichlet([1] * randint(2, 10)) for _ in range(10)])
def test_stochastic(pi):
    counts = defaultdict(int)
    trials = 20000

    for _ in range(trials):
        action = stochastic_policy(pi)[0] # Log action not parameters
        counts[action] += 1

    for action, count in counts.items():
        assert abs(count / trials - pi[action]) < probability_fudge_factor, (
               'should be around expected probability')


def test_stochastic_fails():
    # should assert non empty policy
    with pytest.raises(AssertionError):
        stochastic_policy([])

    # should all be non negative
    with pytest.raises(ValueError):
        stochastic_policy([-0.1, 0.9])

    # need to sum to 1
    with pytest.raises(ValueError):
        stochastic_policy([0.1, 0.9, 0.2])


##################
# Uniform Random #
##################
@pytest.mark.parametrize('length', [
    10,
    5,
    2,
    100,
    ])
def test_uniform_random(length):
    counts = defaultdict(int)
    trials = 20000

    for _ in range(trials):
        action = uniform_random_policy(length)[0] # Log action not parameters
        counts[action] += 1

    for action, count in counts.items():
        assert abs(count / trials - 1/length) < probability_fudge_factor, (
               'should be around expected probability')


def test_uniform_random_fails():
    with pytest.raises(ValueError):
        uniform_random_policy(0)

    with pytest.raises(ValueError):
        uniform_random_policy(-1)


##################
# Epsilon Greedy #
##################
@pytest.mark.parametrize('values, epsilon, max_idx', [
    ([100, 300, 200, 300], 0.2, [1, 3]),
    ([100, 300, 10, 100], 0.9, [1]),
    ([100, 300, 200, 100], 0.01, [1]),
    ([100, 300, 200, 100], 0.0, [1]),
    ([100, 300, 200, 300], 1.0, [1, 3]),
    ])
def test_epsilon_greedy(values, epsilon, max_idx):
    values = [100, 300, 200, 300]
    epsilon = 0.2
    max_idx = [1, 3]

    counts = defaultdict(int)
    trials = 20000

    for _ in range(trials):
        action = epsilon_greedy(epsilon, values)[0] # Log action not parameters
        print(action)
        counts[action] += 1

    random_prob = 1 / len(values) * epsilon
    max_prob = (1 - epsilon) / len(max_idx) + random_prob
    for action, count in counts.items():
        if action in max_idx:
            assert abs(count / trials - max_prob) < probability_fudge_factor, (
                   'should be around expected probability')


def test_epsilon_greedy_fails():
    # epsilon should be positive
    with pytest.raises(AssertionError):
        epsilon_greedy(-0.01, [100, 200])

    # epsilon should be less than or equal to 1
    with pytest.raises(AssertionError):
        epsilon_greedy(1.01, [100, 200])
