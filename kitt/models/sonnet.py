from typing import List, Generator, Tuple

import tensorflow as tf
import sonnet as snt
import numpy as np
import pandas as pd

from kitt.types import ModelType, ModelPolicyType, PolicyType


##############$##
# Preset Models #
#################
class DQNModel(snt.Module):
    def __init__(self, name='DQNNature', output_size=2):
        super(DQNModel, self).__init__(name=name)

        self._output_size = output_size

        self._conv_1 =  snt.Conv2D(output_channels=16,
                                   kernel_shape=(8, 8),
                                   stride=(1, 1),
                                   name="Conv1")
        self._conv_2 =  snt.Conv2D(output_channels=32,
                                   kernel_shape=(4, 4),
                                   stride=(1, 1),
                                   name="Conv2")
        self._flatten = snt.Flatten()
        self._fc_1 = snt.Linear(256, name="FC")
        self._output_layer = snt.Linear(output_size, name="OutputFC")

    def __call__(self, x):
        h = tf.nn.relu(self._conv_1(x))
        h = tf.nn.relu(self._conv_2(h))
        h = self._flatten(h)
        h = tf.nn.relu(self._fc_1(h))
        return self._output_layer(h)


class MultiInputModel(snt.Module):
    def __init__(self, models, name="MulitInput"):
        super(MultiInputModel, self).__init__(name=name)
        self._models = models

    @property
    def trainable_variables(self):
        return sum(map(lambda x: x.trainable_variables(), self._models))

    @property
    def variables(self):
        return sum(map(lambda x: x.variables(), self._models))

    def __call__(self, xs):
        model_outputs = []
        for i, model in enumerate(self._models):
            model_outputs.append(model(xs[i]))
        return tf.concat(model_outputs, 1)


class IterationModel(snt.Module):
    def __init__(self, model, name="Iteration", output_size=1, iter_size=5, activation=None):
        super(IterationModel, self).__init__(name=name)
        self._model = model
        self._iter_size = iter_size
        self._output_size = output_size
        self._activation = activation

    @property
    def trainable_variables(self):
        return self._model.trainable_variables

    def __call__(self, x):
        # split each element of x into iterations
        x_shape = tf.shape(x)
        x = tf.reshape(x, [-1, self._iter_size])

        # run through model
        output = tf.reshape(self._model(x),
                            [x_shape[0], -1])
        
        if self._activation is not None:
            output = self._activation(output)

        # return logits
        return output


##################
# Policy Wrapper #
##################
def sonnet_policy(model: ModelType,
                  policy: PolicyType,
                  policy_params: pd.Series = pd.Series()) -> Generator[Tuple[ModelPolicyType, List],
                                                                       None,
                                                                       None]:
    while True:
        p_params = list(map(lambda value: next(value), policy_params.values))
        epoch_params = pd.Series(p_params, index=policy_params.index)

        def out_policy(state):
            # TODO (ratclife@dino.ai): sort out the dimensions
            logits = model(tf.expand_dims(state, 0))[0]
            return policy(*p_params, logits)[0], pd.Series()

        yield out_policy, epoch_params

def batch_sonnet_policy(model,
                        policy,
                        policy_params=pd.Series()):
    while True:
        p_params = list(map(lambda value: next(value), policy_params.values))
        epoch_params = pd.Series(p_params, index=policy_params.index)

        @tf.function
        def out_policy(states):
            # TODO (ratclife@dino.ai): sort out the dimensions
            logits_batch = model(tf.stack(states))
            return policy(*p_params, logits_batch)[0], {}

        yield out_policy, epoch_params


##############
# Train Step #
##############
@tf.function
def sonnet_optimiser(trainable_variables, optimiser, lr, loss_fn, params, max_grad_norm=None, grad_clip=None):
    with tf.GradientTape() as tape:
        loss = 0
        loss += loss_fn(*params)

    # TODO (ratclife@dino.ai): remove duplicate variables
    train_vars = trainable_variables()
    grads = tape.gradient(loss, train_vars)

    # max grad norm
    if max_grad_norm is not None:
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

    # max grad value
    if grad_clip is not None:
        grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]

    optimiser.learning_rate.assign(lr)
    optimiser.apply(grads, train_vars)

    return grads, lr, loss


########################
# Parameter Operations #
########################
def sonnet_set_parameters(model, parameters):
    # should assert that paramters is numpy array, plan on being able to pass across processes
    for var in model.trainable_variables:
        current_vars = parameters[:np.prod(var.shape)].reshape(var.shape)
        var.assign(current_vars)
        parameters = parameters[np.prod(var.shape):]
    return model

def sonnet_get_parameters(model):
    return np.concatenate(list(map(lambda var: var.numpy().flatten(), model.trainable_variables)))

#########
# UTILS #
#########
class SequentialRecurrent(snt.Module):
    def __init__(self,
                 layers,
                 name = None):
        super(SequentialRecurrent, self).__init__(name=name)
        self._layers = list(layers) if layers is not None else []


    def __call__(self, inputs, memory_states=None):
        current_memory = 0;
        final_states = []
        outputs = inputs
        for mod in self._layers:
            if hasattr(mod, 'initial_state'):
                state = mod.initial_state(tf.shape(outputs)[0])
                if memory_states is not None:
                    state = memory_states[current_memory]
                    current_memory += 1

                if isinstance(mod, SequentialRecurrent):
                    outputs, output_state = mod(outputs, state)
                else:
                    outputs = tf.transpose(outputs, [1, 0, 2])
                    outputs, output_state = snt.dynamic_unroll(mod, outputs, state)
                    outputs = tf.transpose(outputs, [1, 0, 2])

                final_states.append(output_state)
            else:
                outputs = snt.BatchApply(mod)(outputs)

        return outputs, final_states


    def initial_state(self, batch_size):
        initial_states = []
        for mod in self._layers:
            if hasattr(mod, 'initial_state'):
                initial_states.append(mod.initial_state(batch_size))
        return initial_states

class SharedTrunk(snt.Module):
    def __init__(self,
                 trunk,
                 heads,
                 name = None):
        super(SharedTrunk, self).__init__(name=name)
        self._trunk = trunk
        self._heads = list(heads) if heads is not None else []

    def __call__(self, inputs):
        trunk_out = self._trunk(inputs)
        return [head(trunk_out) for head in self._heads]


class SharedTrunkRecurrent(snt.Module):
    def __init__(self,
                 trunk,
                 heads,
                 name = None):
        super(SharedTrunkRecurrent, self).__init__(name=name)
        self._trunk = trunk
        self._heads = list(heads) if heads is not None else []

    def __call__(self, inputs, memory_states=None):
        trunk_out, final_memory = self._trunk(inputs, memory_states)
        return [head(trunk_out) for head in self._heads] + [final_memory]

    def initial_state(self, batch_size):
        return self._trunk.initial_state(batch_size)
