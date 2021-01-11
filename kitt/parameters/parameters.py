from typing import Iterator
from itertools import repeat

from kitt.types import T

def constant(value:T) -> Iterator[T]:
    return repeat(value)

def repeat_parameter(repeat_n, parameter_generator):
    n = 0
    value = next(parameter_generator)
    while True:
        if n == repeat_n:
            n = 0
            value = next(parameter_generator)
        n+=1
        yield value

def linear(from_value:float, to_value:float, steps:int) -> Iterator[float]:
    current = from_value
    step = (from_value - to_value) / steps
    yield current
    while True:
        if (from_value > to_value and current > to_value) or \
           (from_value < to_value and current < to_value):
            current -= step

        # set to proper value when we get within half a step
        if (abs(current - to_value) < abs(step/2)):
            current = to_value

        yield current
