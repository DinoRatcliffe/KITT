from itertools import chain

import sonnet as snt
import tensorflow as tf
import pandas as pd
import numpy as np

from kitt.utils import take, epocher
from kitt.parameters import constant
from tensorflow_probability import distributions as tfd


##################
# Hyperparamters #
##################
def vae_hyperparameters(kl_tolerance=constant(0.5),
                        epoch_data_size=constant(100),
                        batch_size=constant(32),
                        beta=constant(1.0)):
    while True:
        yield pd.Series([next(kl_tolerance),
                         next(epoch_data_size),
                         next(batch_size),
                         next(beta)],
                        index=['kl_tolerance',
                               'epoch_data_size',
                               'batch_size',
                               'beta'])


##########
# Losses #
##########
def vae_loss(encoder, decoder, kl_tolerance, beta, inputs):
    return kl_loss(encoder(inputs), kl_tolerance) * beta + reconstruction_loss(decoder(encoder(inputs)), inputs)

def reconstruction_loss(decoder_output, inputs):
    if isinstance(decoder_output, tfd.Distribution):
        # Get negative log likeligood of distribution 
        return tf.reduce_mean(-decoder_output.log_prob(inputs))
    else:
        # Mean Squared Error
        return tf.reduce_mean(tf.reduce_sum((inputs - decoder_output) ** 2, axis=[1, 2, 3]))

def kl_loss(encoder_output, kl_tolerance):
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(tf.shape(encoder_output)[1], dtype=tf.float64), scale=tf.ones(tf.shape(encoder_output)[1], dtype=tf.float64)), reinterpreted_batch_ndims=1)
#    prior = tfd.MultivariateNormalTriL(loc=tf.zeros(tf.shape(encoder_output)[1], dtype=tf.float64))
    kl = tfd.kl_divergence(encoder_output, prior)
    kl = tf.maximum(kl, kl_tolerance * encoder_output.shape[1])

    return tf.reduce_mean(kl)


#############
# Algorithm #
#############
def vae(hyperparams, image_generator, epoch):
    # Inintial setup
    encoder = epoch.vae_state.encoder
    decoder = epoch.vae_state.decoder
    hyperparameters = next(hyperparams)

    new_epoch = epoch.copy()
    new_epoch['vae_state']['hyperparameters'] = hyperparameters

    # Produce inputs
    inputs = tf.constant(np.array(list(take(hyperparameters.epoch_data_size, image_generator))), tf.float64)

    # turn pairs into batches
    batch_data = [inputs]
    append_data = lambda x: list(map(lambda y: [encoder, decoder, hyperparameters.kl_tolerance, hyperparameters.beta] + y, x))

    batches = map(append_data, epocher(int(hyperparameters.batch_size), batch_data))
    batches = list(chain(*take(1, batches)))

    inputs = inputs[0:5]
    encoder_output = encoder(inputs)
    decoder_output = decoder(encoder_output)
    kl_losses = kl_loss(encoder_output, hyperparameters.kl_tolerance)
    reconstruction_losses = reconstruction_loss(decoder_output, inputs)
    vae_losses = vae_loss(encoder, decoder, hyperparameters.kl_tolerance, hyperparameters.beta, inputs)

    decoder_sample = tf.concat((inputs[0:3], decoder_output[0:3]), axis=2)

    # calculate loss
    new_epoch['vae_state']['log_data'] = pd.Series([vae_losses,
                                                    kl_losses,
                                                    reconstruction_losses,
                                                    decoder_sample],
                                                   index=['loss',
                                                          'kl_loss',
                                                          'reconstruction_loss',
                                                          'decoder_sample'])

    return new_epoch, vae_loss, batches
