from typing import Iterator
from functools import reduce
from itertools import accumulate

import numpy as np

from kitt import reverse

# these should probably expect numpy arrays


def episodic_returns(rewards:Iterator[float]) -> Iterator[float]:
    return discounted_returns(rewards, 1.0)


def discounted_returns(rewards:Iterator[float], discount:float) -> Iterator[float]:
    assert len(rewards) > 0, ('trying to pass empty list as rewards')
    assert 0 <= discount <= 1, ('discount should be between 0 and 1')
    return reverse(list(accumulate(reverse(list(rewards)), lambda r, rt: rt + r * discount)))


def advantage(rewards, values, terminals, last_value, gamma, lam):
    advantages = []
    lastgaelam = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 0.0 if terminals[-1] else 1.0
            nextvalue = last_value
        else:
            nextnonterminal = 0.0 if terminals[t+1] else 1.0
            nextvalue = values[t+1]

        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages.append(lastgaelam)

    return reverse(advantages)


def gae_lambda_advantage(gamma, lam, rewards, values, last_value=0):
    rewards = np.append(rewards, last_value)
    values = np.append(values, last_value)
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    return discounted_returns(deltas, gamma * lam)
