from typing import Sequence, Union, Callable, Tuple, Dict
from collections import namedtuple
from random import sample

from numpy import argmax, argwhere, amax, random

def multi_policy(policies: Sequence[Sequence[float]], policy_fn):
    return [policy_fn(p) for p in policies]


def stochastic_policy(policy: Sequence[float]) -> Tuple[int, Dict]:
    assert len(policy) > 0, ('passing in empty policy')

    # @speed this function is slow, should speed up in future
    return random.choice(range(len(policy)), p=policy), {}


def greedy_policy(values: Sequence[float]) -> Tuple[int, Dict]:
    # We should randomly break ties
    maxes = argwhere(values == amax(values)).flatten().tolist()
    return int(sample(maxes, 1)[0]), {}


def uniform_random_policy(length: int) -> Tuple[int, Dict]:
    return random.randint(length), {}


def epsilon_greedy(epsilon: float, values: Sequence[float]) -> Tuple[int, Dict]:
    assert 0 <= epsilon <= 1, (
           'epsilon needs to be a positive value between 0 and 1')
    assert len(values) > 0, ('passing an empty sequence')

    greedy = random.random() > epsilon
    return greedy_policy(values)[0] if greedy else uniform_random_policy(len(values))[0], {}
