from datetime import datetime
from functools import partial
import os

import sonnet as snt
import tensorflow as tf
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tensorflow_probability import layers as tfpl

from kitt.utils import iterate, time, increment_epoch, compose, take, epoch_interval, single_trainer
from kitt.policy_gradient import ppo_hyperparams, ppo
from kitt.policies import stochastic_policy
from kitt.models import sonnet_policy, sonnet_optimiser, SharedTrunk, batch_sonnet_policy
from kitt.parameters import constant, linear
from kitt.environments import cartpole_parameterised
from kitt.evaluation import evaluate_policy
from kitt.experiance_generators import episodes_generator, rollout_generator, multiprocess_generator
from kitt.loggers import CSVWriter, csv_log, tensorboard_log

def save_ppo_model(directory, epoch):
    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float64)])
    def inference(x):
        return epoch.ppo_state.model(x)
    to_save = snt.Module()
    to_save.inference = inference
    to_save.all_variables = epoch.ppo_state.model.variables
    tf.saved_model.save(to_save, f"{directory}/{epoch.epoch}/testing_model")
    return epoch


def train_ppo(outdir, parameter, value, epochs):
    log_dir = os.path.join(outdir, 'logs')
    save_dir = os.path.join(outdir, 'models')

    ####################
    # Hyper Parameters #
    ####################
    rollout_length = constant(128)
    rollouts_per_step = constant(12)
    batch_size = constant(64)
    steps_per_epoch = constant(4)
    entropy_coefficient = constant(0.01)
    value_coefficient = constant(1.0)
    clip_range = constant(0.2)
    gamma = constant(0.98)
    lam = constant(1)
    alpha = constant(1e-3)
    optimiser = snt.optimizers.Adam(tf.Variable(next(alpha)))

    ppo_hyperparameters = ppo_hyperparams(rollout_length,
                                          rollouts_per_step,
                                          batch_size,
                                          steps_per_epoch,
                                          entropy_coefficient,
                                          value_coefficient,
                                          clip_range,
                                          gamma,
                                          lam)


    # ENV
    initial_states, transition = cartpole_parameterised(parameter, value)
    eval_initial_states, eval_transition = cartpole_parameterised(parameter, value)

    # Model
    trunk = snt.Sequential([snt.Linear(40),
                            tf.nn.relu,
                            snt.Linear(35),
                            tf.nn.relu,
                            snt.Linear(30),
                            tf.nn.relu])
    policy_output = snt.Linear(2)
    value_output = snt.Linear(1)

    model = SharedTrunk(trunk, [policy_output, value_output])
    policy_dist = tfpl.OneHotCategorical(2, dtype=tf.float64)

    policy = sonnet_policy(lambda x: policy_dist(model(x)[0]),
                           stochastic_policy)

    # Has to be function at the moment due to sonnet limitation
    trainable_variables = lambda: model.trainable_variables

    # setup initial state
    ppo_state = pd.Series([model, policy_dist], index=['model', 'dist'])
    initial_state = pd.Series([0, ppo_state, datetime.now()],
                              index=['epoch', 'ppo_state', 'walltime'])

    # ppo_training operation
    rollout_gen = rollout_generator(initial_states, transition, next(policy)[0], next(rollout_length))
    ppo_optimisation = partial(single_trainer, partial(sonnet_optimiser,
                                                       trainable_variables,
                                                       optimiser),
                                               alpha,
                                               partial(ppo, ppo_hyperparameters,
                                                            rollout_gen))
    # eval and save
    writer = tf.summary.create_file_writer(log_dir)
    csv_writer = CSVWriter(log_dir)
    eval_and_save_model = compose(partial(tensorboard_log,
                                          writer,
                                          ["environment_evaluation"]),
                                  partial(csv_log,
                                          csv_writer,
                                          ["environment_evaluation"]),
                                  partial(evaluate_policy,
                                          100,
                                          eval_initial_states,
                                          eval_transition,
                                          next(policy)[0]),
                                  partial(save_ppo_model,
                                          save_dir))

    # main loop
    train_loop = compose(partial(epoch_interval, 50,
                                                 eval_and_save_model),
                         partial(tensorboard_log,
                                 writer,
                                 ["ppo_state", "log_data"]),
                         partial(csv_log,
                                 csv_writer,
                                 ["ppo_state", "log_data"]),
                         increment_epoch,
                         ppo_optimisation)

    # run training
    print(list(take(epochs + 1, iterate(train_loop, initial_state))))

if __name__ == '__main__':
    arg_parser = ArgumentParser(description='Train PPO agent with set cartpole parameters')
    arg_parser.add_argument('-o', '--outdir', type=str, default='tmp/ppo-cartpole-test',
            help='The directory to output the saved models and logs')
    arg_parser.add_argument('-p', '--parameter', type=str, default='masscart',
            help='The parameter to set')
    arg_parser.add_argument('-v', '--value', type=float, default=1.0,
            help='The value for the above parameter to set')
    arg_parser.add_argument('-e', '--epochs', type=int, default=600,
            help='The number of epochs to run for')
    args = arg_parser.parse_args()

    train_ppo(args.outdir, args.parameter, args.value, args.epochs)
