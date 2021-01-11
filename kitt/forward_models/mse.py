from itertools import chain

import tensorflow as tf
import pandas as pd
import numpy as np

from kitt.parameters import constant
from kitt.utils import take, one_hot, epocher

##################
# Hyperparamters #
##################
def mse_forward_model_hyperparameters(rollout_length=constant(128),
                                      rollouts_per_step=constant(12),
                                      batch_size=constant(32),
                                      reward_weight=constant(1.0),
                                      terminal_weight=constant(1.0)):
    while True:
        yield pd.Series([next(rollout_length),
                         next(rollouts_per_step),
                         next(batch_size),
                         next(reward_weight),
                         next(terminal_weight)],
                        index=['rollout_length',
                               'rollouts_per_step',
                               'batch_size',
                               'reward_weight',
                               'terminal_weight'])
########
# Loss #
########
def mean_squared_error(forward_model, predicted_observation, observation, observation_prime):
    observation_prime_diff = observation_prime - observation

    if hasattr(forward_model, 'initial_state'):
        observation_prime_diff = observation_prime_diff[:, :, :tf.shape(predicted_observation)[-1]]
        loss = tf.reduce_sum(tf.reduce_mean((predicted_observation - observation_prime_diff) ** 2, -1), 1)
    else:
        observation_prime_diff = observation_prime_diff[:, :tf.shape(predicted_observation)[-1]]
        loss = (predicted_observation - observation_prime_diff) ** 2

    return tf.reduce_mean(loss)

def reward_error(forward_model, predicted_reward, reward):
    predicted_reward = tf.squeeze(predicted_reward)
    return tf.reduce_mean((predicted_reward - reward) ** 2)

def terminal_error(forward_model, predicted_terminal, terminal):
    terminal = tf.cast(terminal, tf.int32)
    return tf.reduce_mean(-predicted_terminal.log_prob(terminal))

def full_loss(forward_model, reward_weight, terminal_weight, observation, action, observation_prime, rewards, terminals):
    one_hot_actions = tf.one_hot(tf.cast(action, tf.int32), 2, dtype=tf.float64)
    x = tf.concat((observation, one_hot_actions), axis=-1)

    if hasattr(forward_model, 'initial_state'):
        predicted_observation, predicted_reward, predicted_terminal, _ = forward_model(x)
    else:
        predicted_observation, predicted_reward, predicted_terminal = forward_model(x)

    loss = mean_squared_error(forward_model, predicted_observation, observation, observation_prime)

    if reward_weight > 0.0:
        loss += reward_error(forward_model, predicted_reward, rewards) * reward_weight

    if terminal_weight > 0.0:
        loss += terminal_error(forward_model, predicted_terminal, terminals) * terminal_weight

    return loss

#############
# Algorithm #
#############
def mse_forward_model(hyperparams, rollout_generator, epoch, encoder=None):
    if encoder is not None:
        print(encoder)

    # Initial Setup
    forward_model = epoch.mse_state.forward_model

    hyperparameters = next(hyperparams)

    new_epoch = epoch.copy()
    new_epoch['mse_state']['hyperparameters'] = hyperparameters

    # Produce Experiances
    rollouts = take(int(hyperparameters.rollouts_per_step), rollout_generator)

    # Create Batches
    if hasattr(forward_model, 'initial_state'):
        experiances = list(rollouts)
        observations = tf.constant(list(map(lambda exp: exp.observation.values, experiances)), tf.float64)
        actions = tf.constant(list(map(lambda exp: exp.action.values, experiances)), tf.float64)
        observation_primes = tf.constant(list(map(lambda exp: exp.observation_prime.values, experiances)), tf.float64)
        rewards = tf.constant(list(map(lambda exp: exp.reward.values, experiances)), tf.float64)
        terminals = tf.constant(list(map(lambda exp: exp.terminal.values, experiances)), tf.bool)
    else:
        experiances = pd.concat(list(rollouts))
        observations = tf.constant(list(experiances.observation.values), tf.float64)
        actions = tf.constant(list(experiances.action.values), tf.int32)
        observation_primes = tf.constant(list(experiances.observation_prime.values), tf.float64)
        rewards = tf.constant(list(experiances.reward.values), tf.float64)
        terminals = tf.constant(list(experiances.terminal.values), tf.bool)

    if encoder is not None:
        observation_primes = encoder(observation_primes)
        observations = encoder(observations)
        

    batch_data = [observations, actions, observation_primes, rewards, terminals]
    append_data = lambda x: list(map(lambda y: [forward_model, hyperparameters.reward_weight, hyperparameters.terminal_weight] + y, x))

    batches = map(append_data, epocher(int(hyperparameters.batch_size), batch_data))
    batches = list(chain(*take(1, batches)))
    loss = full_loss(forward_model, hyperparameters.reward_weight, hyperparameters.terminal_weight, observations, actions, observation_primes, rewards, terminals)

    one_hot_actions = tf.one_hot(tf.cast(actions, tf.int32), 2, dtype=tf.float64)
    x = tf.concat((observations, one_hot_actions), axis=-1)

    if hasattr(forward_model, 'initial_state'):
        predicted_observation, predicted_reward, predicted_terminal, _ = forward_model(x)
    else:
        predicted_observation, predicted_reward, predicted_terminal = forward_model(x)

    reward_loss = tf.constant(0, tf.float64)
    if hyperparameters.reward_weight != 0.0:
        reward_loss = reward_error(forward_model, predicted_reward, rewards)

    terminal_loss = tf.constant(0, tf.float64)
    if hyperparameters.reward_weight != 0.0:
        terminal_loss = terminal_error(forward_model, predicted_terminal, terminals)

    # calculate loss
    new_epoch['mse_state']['log_data'] = pd.Series([loss,
                                                    reward_loss,
                                                    terminal_loss],
                                                   index=['loss',
                                                          'reward-loss',
                                                          'terminal-loss'])


    return new_epoch, full_loss, batches
