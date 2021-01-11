import pytest

from kitpy.returns import episodic_returns, discounted_returns


####################
# Episodic Returns #
####################
@pytest.mark.parametrize("rewards, returns", [
    ([0, 1, 2, 2], [5, 5, 4, 2]),
    ([0, 1, 2, -2], [1, 1, 0, -2]),
    ([0, -1, 2, -2], [-1, -1, 0, -2]),
    ([0], [0])
    ])
def test_episodic_returns_successes(rewards, returns):
    assert episodic_returns(rewards) == returns, (
           'should return list with rewards fully summed at each timestep')


def test_episodic_returns_failures():
    # Throw exception when given empty list of rewards
    with pytest.raises(AssertionError):
        episodic_returns([])


######################
# Discounted Returns #
######################
@pytest.mark.parametrize("rewards, discount, returns", [
    ([0, 1, 2, 2], 0.5, [1.25, 2.5, 3, 2]),
    ([0, 10, 2, 2], 0.5, [5.75, 11.5, 3, 2]),
    ([3, 10, 0, 0], 0.5, [8, 10, 0, 0]),
    ([0, 1, 2, 2], 1.0, [5, 5, 4, 2]),
    ([0, 1, 2, -2], 0.5, [0.75, 1.5, 1, -2]),
    ([0, -1, 2, -2], 0.5, [-0.25, -0.5, 1, -2]),
    ([0, 1, 2, 2], 0.0, [0, 1, 2, 2]),
    ([0, 1, 2, 2], 0.1, [0.122, 1.22, 2.2, 2]),
    ([0], 0.99, [0]),
    ])
def test_discounted_returns_successes(rewards, discount, returns):
    assert discounted_returns(rewards, discount) == returns, (
           'should return list with rewards summed at each timestep without '
           'discounted future rewards')


def test_discounted_returns_failures():
    # should throw exception when given empty list for rewards
    with pytest.raises(AssertionError):
        discounted_returns([], 0)

    # should throw exception when discount < 0
    with pytest.raises(AssertionError):
        discounted_returns([0], -0.1)

    # should throw exception when discount > 1
    with pytest.raises(AssertionError):
        discounted_returns([0], 1.1)
