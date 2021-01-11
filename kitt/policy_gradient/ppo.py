from random import shuffle
from functools import partial
from itertools import chain

import pandas as pd
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
import numpy as np

from kitt.experiance_generators import rollout_generator
from kitt.utils import append, compose, split_on_terminal, entropy, epocher
from kitt.returns import advantage, gae_lambda_advantage, discounted_returns
from kitt.parameters import constant
from kitt import take


##################
# Hyperparamters #
##################
def ppo_hyperparams(rollout_length=constant(10),
                    rollouts_per_step=constant(5),
                    batch_size=constant(32),
                    steps_per_epoch=constant(4),
                    entropy_coefficient=constant(0.01),
                    value_coefficient=constant(0.5),
                    clip_range=constant(0.1),
                    gamma=constant(0.99),
                    lam=constant(0.95),
                    clip_advantage=constant(False)):
    """PPO Hyperparameters

    TODO (ratcliffe@dino.ai): give explanation of each parameter
    """
    while True:
        yield pd.Series([next(rollout_length),
                         next(rollouts_per_step),
                         next(batch_size),
                         next(steps_per_epoch),
                         next(entropy_coefficient),
                         next(value_coefficient),
                         next(clip_range),
                         next(gamma),
                         next(lam),
                         next(clip_advantage)],
                        index=['rollout_length',
                               'rollouts_per_step',
                               'batch_size',
                               'steps_per_epoch',
                               'entropy_coefficient',
                               'value_coefficient',
                               'clip_range',
                               'gamma',
                               'lam',
                               'clip_advantage'])


########
# Loss #
########
def ppo_value_loss(value_prediction, old_value, returns, cliprange):
    value_prediction_clipped = old_value + tf.clip_by_value(value_prediction - old_value, 
                                                            -cliprange,
                                                            cliprange)

    value_loss_1 = tf.square(value_prediction - returns)
    value_loss_2 = tf.square(value_prediction_clipped - returns)

    return .5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

def ppo_policy_loss(dist,
                    policy_distribution,
                    old_policy_distribution,
                    actions,
                    advantages,
                    cliprange):
    policy_distribution = dist(policy_distribution)
    old_dist = dist(old_policy_distribution)

    neg_log_p_a = -policy_distribution.log_prob(actions)
    old_neg_log_p_a = -old_dist.log_prob(actions)

    ratio = tf.exp(old_neg_log_p_a - neg_log_p_a)

    policy_loss_1 = -advantages * ratio
    policy_loss_2 = -advantages * tf.clip_by_value(ratio, 
                                                   1.0 - cliprange,
                                                   1.0 + cliprange)

    return tf.reduce_mean(tf.maximum(policy_loss_1, policy_loss_2))

@tf.function
def ppo_loss(entropy_coefficient,
             value_coefficient,
             cliprange,
             model,
             dist,
             states,
             actions,
             old_policy_distribution,
             old_value,
             advantages,
             returns):

    policy_distribution, value_prediction = model(states)

    # Value loss
    l_value = ppo_value_loss(value_prediction, old_value, returns, cliprange)

    # Policy loss
    l_policy = ppo_policy_loss(dist,
                               policy_distribution,
                               old_policy_distribution,
                               actions,
                               advantages,
                               cliprange)

    # entropy
    if entropy_coefficient > 0:
        ent = tf.reduce_mean(dist(policy_distribution).entropy())
    else:
        ent = 0.0

    # Combined loss
    return l_policy - (ent * entropy_coefficient) + (l_value * value_coefficient)


def set_final_value(model, rollout):
    if not rollout.terminal.iloc[-1]:
        state_value = model(tf.constant([rollout.observation.iloc[-1]]))[1].numpy()[0,0]

        reward = rollout.reward.iloc[-1]
        rollout.reward.iloc[-1] = reward + state_value 

    return rollout

def discount_within_rollouts(rollout, gamma):
    terminals = rollout.terminal.values

    size = len(terminals)
    idx_terminal = [idx + 1 for idx, val in
                    enumerate(terminals) if val] 

    if (len(idx_terminal) > 0):
        rewards = [rollout.reward.values[i: j] for i, j in
                    zip([0] + idx_terminal, idx_terminal +
                    ([size] if idx_terminal[-1] != size else []))]
    else:
        rewards = [rollout.reward.values]

    discounted_rewards = np.concatenate(list(map(lambda r: discounted_returns(r, gamma), rewards)))

    return discounted_rewards

def split_on_terminal(rollout):
    terminals = rollout.terminal.values
    size = len(terminals)
    idx_terminal = [idx + 1 for idx, val in
                    enumerate(terminals) if val]

    if (len(idx_terminal) > 0):
        rollouts = [rollout[i:j] for i, j in
                    zip([0] + idx_terminal, idx_terminal +
                    ([size] if idx_terminal[-1] != size else[]))]
    else:
        rollouts = [rollout]

    return rollouts


def append_policy_value(model, rollout):
    policy, value = model(tf.constant(list(rollout.observation.values)))
    rollout = append('estimated_value', list(np.squeeze(value.numpy())), rollout)
    rollout = append('policy', list(np.squeeze(policy.numpy())), rollout)
    return rollout

#############
# Algorithm #
#############
def ppo(hyperparameters,
        rollout_gen,
        epoch):
    """Proximal Policy Optimisation

    TODO (ratcliffe@dino.ai): Explain PPO here
    """

    # 1. Intitial setup
    model = epoch.ppo_state.model
    dist = epoch.ppo_state.dist
    hyperparams = next(hyperparameters)

    new_epoch = epoch.copy()
    new_epoch['ppo_state']['hyperparameters'] = hyperparams

    # 3. Collect set of trajectories
    rollouts = take(hyperparams.rollouts_per_step, rollout_gen)
    
    # 5a. Estimate values
    rollouts = map(partial(append_policy_value, model), rollouts)

    # 5b. Compute GAE-Lambda Advantage
    gae_lambda = partial(gae_lambda_advantage, hyperparams.gamma, hyperparams.lam)
    append_gae_lambda = lambda rollout: append("gae_lambda_advantage",
                                               gae_lambda(rollout.reward, rollout.estimated_value),
                                               rollout)

    handle_split = lambda split: append("gae_lambda_advantage",
                                        gae_lambda(split.reward,
                                                   split.estimated_value,
                                                   0 if split.terminal.iloc[-1] else split.estimated_value.iloc[-1]),
                                        split)

    rollouts = map(compose(pd.concat,
                           partial(map, handle_split),
                           split_on_terminal), rollouts) # list of lists of dataframes

    rollouts = pd.concat(list(rollouts))
    states = tf.constant(np.array(list(rollouts.observation.values)))

    old_policy_distribution = tf.constant(model(states)[0])
    old_value = tf.constant(np.array(list(rollouts.estimated_value.values)))

    actions = tf.constant(np.array(list(rollouts.action.values)))
    actions = tf.one_hot(tf.cast(actions, tf.int32), old_policy_distribution.shape[1])

    advs = np.array(list(rollouts.gae_lambda_advantage.values))
    returns = old_value + advs
    advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)

    if hyperparams.clip_advantage:
        advantages = tf.constant(np.clip(advs, a_min=0, a_max=None))
    else:
        advantages = tf.constant(advs)


    batch_data = [states, actions, old_policy_distribution, old_value, advantages, returns]

    # 6. Generate b batches of n full epochs
    append_data = lambda x: list(map(lambda y: [hyperparams.entropy_coefficient,
                                                hyperparams.value_coefficient,
                                                hyperparams.clip_range,
                                                model,
                                                dist] + y, x))

    batches = map(append_data, epocher(int(hyperparams.batch_size), batch_data))
    batches = list(chain(*take(hyperparams.steps_per_epoch, batches)))

    # Calculate Loss
    batch_loss = 1024
    policy_distribution_raw, values = model(states[:batch_loss])
    policy_distribution = dist(policy_distribution_raw)

    if hyperparams.entropy_coefficient > 0:
        policy_entropy = policy_distribution.entropy()
    else:
        policy_entropy = np.array([0.0])

    policy_mean = policy_distribution.mean()
    policy_variance = policy_distribution.variance()

    v_loss = ppo_value_loss(values[:batch_loss], old_value[:batch_loss], returns[:batch_loss], hyperparams.clip_range)
    p_loss = ppo_policy_loss(dist,
                             policy_distribution_raw,
                             old_policy_distribution[:batch_loss],
                             actions[:batch_loss],
                             advantages[:batch_loss],
                             hyperparams.clip_range)
    full_loss = ppo_loss(hyperparams.entropy_coefficient,
                         hyperparams.value_coefficient,
                         hyperparams.clip_range,
                         model,
                         dist,
                         states[:batch_loss],
                         actions[:batch_loss],
                         old_policy_distribution[:batch_loss],
                         old_value[:batch_loss],
                         advantages[:batch_loss],
                         returns[:batch_loss])

    # Add log data to current epoch
    new_epoch['ppo_state']['log_data']  = pd.Series([policy_entropy,
                                                     policy_mean,
                                                     policy_variance,
                                                     tf.squeeze(values),
                                                     v_loss,
                                                     p_loss,
                                                     full_loss], index=['policy_entropy',
                                                                        'policy_mean',
                                                                        'policy_variance',
                                                                        'value_prediction',
                                                                        'value_loss',
                                                                        'policy_loss',
                                                                        'full_loss'])

    return new_epoch, ppo_loss, batches
