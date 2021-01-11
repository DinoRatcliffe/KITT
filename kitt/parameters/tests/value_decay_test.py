import pytest

from kitpy.parameters import constant, linear


############
# Constant #
############
def test_constant_value():
    const_value = constant(0.9)
    for _ in range(100):
        assert next(const_value) == 0.9, ('The value should not change')


################
# Linear Decay #
################
@pytest.mark.parametrize("from_value, to_value, steps, increment", [
    (10, 0, 10, 1),
    ])
def test_linear_simple_linear_checks_decay(from_value, to_value, steps, increment):
    value = linear(from_value, to_value, steps)

    for i in range(steps):
        assert next(value) == from_value - increment * i, (
               'the value should linearly decay')

    for i in range(steps):
        assert next(value) == to_value, (
               'the value should not change once reached to_value')


@pytest.mark.parametrize("from_value, to_value, steps, increment", [
    (10, 20, 10, 1),
    ])
def test_linear_simple_linear_checks_increase(from_value, to_value, steps, increment):
    value = linear(from_value, to_value, steps)

    for i in range(steps):
        assert next(value) == from_value + increment * i, (
               'the value should linearly decay')

    for i in range(steps):
        assert next(value) == to_value, (
               'the value should not change once reached to_value')


@pytest.mark.parametrize("from_value, to_value, steps", [
    (0.490, 0.2, 1),
    (390, 39, -20),
    (3, 30, 10),
    (1e-3, 1e-5, 12000),
    (1e-5, 1e-2, 12000),
    ])
def test_linear_complex_linear_checks(from_value, to_value, steps):
    value = linear(from_value, to_value, steps)

    assert next(value) == from_value, 'starts at from_value'
    for i in range(steps-1):
        # other tests try to asses the linearity of the function
        current_value = next(value)
        assert current_value != from_value and current_value != to_value, (
               'should decay over steps')

    for i in range(steps):
        assert next(value) == to_value, (
               'the value should not change once reached to_value')
