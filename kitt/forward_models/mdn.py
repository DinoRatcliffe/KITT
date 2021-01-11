from itertools import chain

import sonnet as snt
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow_probability import distributions as tfd

from kitt.parameters import constant
from kitt.experiance_generators import rollout_generator
from kitt.utils import take, one_hot, epocher


###################
# Hyperparameters #
###################
def mdn_forward_model_hyperparameters(rollout_length=constant(129),
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
def negative_loss_likelihood(forward_model, predicted_observation, observation, observation_prime):
    observation_prime_diff = observation_prime
    
    if hasattr(forward_model, 'initial_state'):
        observation_prime_diff = observation_prime_diff[:, :, :tf.shape(predicted_observation)[-1]]
    else:
        observation_prime_diff = observation_prime_diff[:, :tf.shape(predicted_observation)[-1]]

    loss = -tf.reduce_mean(predicted_observation.log_prob(observation_prime_diff))
    return loss


def reward_error(forward_model, predicted_reward, reward):
    predicted_reward = tf.squeeze(predicted_reward)
    return tf.reduce_mean((predicted_reward - reward) ** 2)

def terminal_error(forward_model, predicted_terminal, terminal):
    terminal = tf.cast(terminal, tf.float64)
    if hasattr(forward_model, 'initial_state'):
        terminal = tf.expand_dims(terminal, 2)
    return tf.reduce_mean(-predicted_terminal.log_prob(terminal))

def full_loss(forward_model, reward_weight, terminal_weight, observation, action, observation_prime, rewards, terminals):
    one_hot_actions = tf.one_hot(tf.cast(action, tf.int32), 2, dtype=tf.float64)
    x = tf.concat((observation, one_hot_actions), axis=-1)

    if hasattr(forward_model, 'initial_state'):
        predicted_observation, predicted_reward, predicted_terminal, _ = forward_model(x)
    else:
        predicted_observation, predicted_reward, predicted_terminal = forward_model(x)

    loss = negative_loss_likelihood(forward_model, predicted_observation, observation, observation_prime)

    loss += reward_error(forward_model, predicted_reward, rewards) * reward_weight

    loss += terminal_error(forward_model, predicted_terminal, terminals) * terminal_weight

    return loss


#############
# Algorithm #
#############
def mdn_forward_model(hyperparams, rollout_generator, epoch, encoder=None):
    # Initial Setup
    forward_model = epoch.mdn_state.forward_model
    hyperparameters = next(hyperparams)
    
    new_epoch = epoch.copy()
    new_epoch['mdn_state']['hyperparameters'] = hyperparameters

    # Produce Experiance
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
        actions = tf.constant(np.array(list(experiances.action.values)), tf.int32)
        rewards = tf.constant(np.array(list(experiances.reward.values)), tf.float64)
        terminals = tf.constant(np.array(list(experiances.terminal.values)), tf.bool)
        observation_primes = tf.constant(np.array(list(experiances.observation_prime.values)), tf.float64)
        observations = tf.constant(np.array(list(experiances.observation.values)), tf.float64)

    if encoder is not None:
        if hasattr(forward_model, 'initial_state'):
            observations = snt.BatchApply(encoder)(observations)
            observation_primes = snt.BatchApply(encoder)(observation_primes)
        else:
            observations = encoder(observations)
            observation_primes = encoder(observation_primes)

    batch_data = [observations, actions, observation_primes, rewards, terminals]
    append_data = lambda x: list(map(lambda y: [forward_model, hyperparameters.reward_weight, hyperparameters.terminal_weight] + y, x))

    batches = map(append_data, epocher(int(hyperparameters.batch_size), batch_data))
    batches = list(chain(*take(1, batches)))

    loss = full_loss(forward_model, hyperparameters.reward_weight, hyperparameters.terminal_weight, observations, actions, observation_primes, rewards, terminals)

    one_hot_actions = tf.one_hot(tf.cast(actions[:300], tf.int32), 2, dtype=tf.float64)
    x = tf.concat((observations[:300], one_hot_actions[:300]), axis=-1)

    if hasattr(forward_model, 'initial_state'):
        predicted_observation, predicted_reward, predicted_terminal, _ = forward_model(x)
    else:
        predicted_observation, predicted_reward, predicted_terminal = forward_model(x)

    nll_loss = negative_loss_likelihood(forward_model, predicted_observation, observations[:300], observation_primes[:300])

    reward_loss = tf.constant(0, tf.float64)
    if hyperparameters.reward_weight != 0.0:
        reward_loss = reward_error(forward_model, predicted_reward, rewards[:300])

    terminal_loss = tf.constant(0, tf.float64)
    if hyperparameters.terminal_weight != 0.0:
        terminal_loss = terminal_error(forward_model, predicted_terminal, terminals[:300])

    # calculate loss
    new_epoch['mdn_state']['log_data'] = pd.Series([loss,
                                                    nll_loss,
                                                    reward_loss,
                                                    terminal_loss,
                                                    tf.reshape(predicted_reward, [-1]),
                                                    tf.reshape(predicted_terminal, [-1])],
                                                   index=['loss',
                                                          'negative-log-likelihood',
                                                          'reward-loss',
                                                          'terminal-loss',
                                                          'predicted-reward',
                                                          'predicted-terminal'])

    return new_epoch, full_loss, batches
