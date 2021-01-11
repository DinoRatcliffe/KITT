from kitt.utils import get

import tensorflow as tf
import numpy as np

from .csvwriter import *

def tensorboard_log(writer, keys, epoch):
    log_data = get(keys, epoch)
    with writer.as_default():
        for key, value in log_data.items():
            if len(value.shape) == 0:
                tf.summary.scalar(key, value, epoch.epoch)
            elif len(value.shape) == 1:
                tf.summary.histogram(key, value, epoch.epoch)
                tf.summary.scalar(key+'-mean', np.mean(value), epoch.epoch)
            elif len(value.shape) == 3:
                tf.summary.image(key, [value], epoch.epoch)
            elif len(value.shape) == 4:
                tf.summary.image(key, value, epoch.epoch)
    return epoch

def csv_log(writer, keys, epoch):
    log_data = get(keys, epoch)
    for key, value in log_data.items():
        if len(value.shape) == 0:
            writer.scalar(key, value, epoch.epoch)
        elif len(value.shape) == 1:
            writer.histogram(key, value, epoch.epoch)
            writer.scalar(key+'-mean', np.mean(value), epoch.epoch)
    return epoch
