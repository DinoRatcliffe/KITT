from itertools import count

import pytest
import numpy as np

from kitt import reverse, take, one_hot, take_upto

# TODO: would be nice to have some form of generative testing
###########
# Reverse #
###########
def test_reverse():
    original_list = [1, 2, 3]
    reversed_list = reverse(original_list)

    assert reversed_list is not original_list, (
           'should be defferent object not same list')

    original_list.reverse()
    assert reversed_list == original_list, (
           'should perform the same operation as the in place reverse')


########
# Take #
########
def test_take():
    full_iter = range(100)

    assert list(take(2, full_iter)) == [0, 1], ('should give first two elements')
    assert list(take(0, full_iter)) == [], (
           'should return empty list when taking zero items')

    empty_iter = range(0)
    assert list(take(0, empty_iter)) == [], (
           'should return empty list when taking from empty iterator')
    assert list(take(100, empty_iter)) == [], (
           'should return empty list when taking from empty iterator')

    short_iter = range(2)
    assert list(take(100, short_iter)) == [0, 1], (
           'should take all remaining values when n > iterator length')

    # should raise error when given negative n value
    with pytest.raises(AssertionError):
        list(take(-1, full_iter))


#############
# Take upto #
#############
def test_take_including():
    values = take_upto(lambda x: x==100, count())
    value = list(values)
    assert len(value) == 101, (
           'should return all values up to and including value that fails '
           'predicate')
    assert value[-1] == 100, (
           'should return all values up to and including, '
           'value that fails predicate')


###########
# One Hot #
###########
def test_one_hot():
    assert (one_hot(3, 0) == np.array([1, 0, 0])).all(), (
           'should generate numpy array with correct one hot value')
    assert (one_hot(3, 1) == np.array([0, 1, 0])).all(), (
           'should generate numpy array with correct one hot value')
    assert (one_hot(3, 2) == np.array([0, 0, 1])).all(), (
           'should generate numpy array with correct one hot value')
    with pytest.raises(AssertionError):
        one_hot(0, 0)
    with pytest.raises(AssertionError):
        one_hot(3, 3)

    # negative indexes
    assert (one_hot(3, -1) == np.array([0, 0, 1])).all(), (
           'should allow for negative looping of indexes')
    assert (one_hot(3, -2) == np.array([0, 1, 0])).all(), (
           'should allow for negative looping of indexes')
    assert (one_hot(3, -3) == np.array([1, 0, 0])).all(), (
           'should allow for negative looping of indexes')
    with pytest.raises(AssertionError):
        one_hot(3, -4)
