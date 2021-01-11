from itertools import islice, accumulate, repeat
from typing import Iterator, List, Generator
from datetime import datetime
import random
import time as t

import numpy as np
import tensorflow as tf
import cv2
import imageio

from kitt.types import T, PredicateType


def save_gif(images, outfile):
    imageio.mimsave(outfile, images)


def save_video(images, outfile):
    images = images.astype(np.uint8)
    images = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images])
    video_shape = images.shape
    out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), 10, (video_shape[1], video_shape[2]))
    for frame in images:
        out.write(frame)
    out.release()


def get(keys, dictionary):
    val = dictionary
    for key in keys:
        val = val[key]
    return val


def identity(x):
    return x


def save_image(image, outfile):
    image = image.astype(np.uint8)
    image = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(outfile, image)


def resize_image(img, width, height):
    height, width, channels = img.shape
    cut_n = int((width - height) / 2)
    return cv2.resize(img,
                      dsize=(width, height),
                      iterpolation=cv2.INTER_AREA)


def display_image(img, window_name='KITT'):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(5)


def single_trainer(optimiser, lr, approach, epoch):
    new_epoch, loss_fn, batches = approach(epoch)
    optimiser_data = [optimiser(tf.constant(next(lr), tf.float32), loss_fn, batch) for batch in batches]
    return new_epoch


def entropy(logits):
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)


def split_on_terminal(rollout):
    splits = []
    terminal_indexes = list(rollout[rollout.terminal].index)
    for start, end in zip([-1] + terminal_indexes, terminal_indexes + [-1]):
        if end == -1:
            if start+1 < len(rollout):
                splits.append(rollout.iloc[start+1:])
        else:
            splits.append(rollout.iloc[start+1:end+1])
    return splits


def epocher(batch_size, data):
    while True:
        seed = random.randint(0, 1e+6)
        shuffled_data = []
        for d in data:
            tf.random.set_seed(seed)
            shuffled_data.append(tf.random.shuffle(d))
        batches = []
        while len(shuffled_data[0]) > batch_size:
            batch = []
            for i in range(len(shuffled_data)):
                batch.append(shuffled_data[i][:batch_size])
                shuffled_data[i] = shuffled_data[i][batch_size:]
                
            batches.append(batch)
        yield batches


def epoch_interval(interval, fn, epoch):
    output_epoch = epoch
    if epoch.epoch % interval == 0 or epoch.epoch == 1:
        output_epoch = fn(epoch)
    return output_epoch
    

# do these exist somewhere in python?
def reverse(input_list:List[T]) -> List[T]:
    input_list = input_list.copy()
    input_list.reverse()
    return input_list


def take(n:int, input_iterator:Iterator[T]) -> Iterator[T]:
    n = int(n)
    assert n >= 0, ('n should be a positive interger')
    return islice(input_iterator, n)


def train(n:int, input_iterator:Iterator[T]) -> T:
    assert n >= 0, ('n should be a positive interger')
    for _ in range(n):
        value = next(input_iterator)
    return value


def take_upto(predicate: PredicateType,
              iterable: Iterator[T]) -> Generator[T, None, None]:
    for x in iterable:
        yield x
        if predicate(x):
            break;


def train_till(predicate: PredicateType,
               iterable: Iterator[T]) -> T:
    for x in iterable:
        if predicate(x):
            return x
        last_item = x
    return last_item


def one_hot(length:int, hot_index:int) -> np.array:
    assert length > 0, ('length should be a non zero positive integer')
    assert -length <= hot_index < length,(
           'hot_index should be a subscriptable value for array of given '
           'length')
    return np.eye(length)[hot_index]


def compose(*funcs):
    def run(x):
        for func in reversed(funcs):
            x = func(x)
        return x
    return run


def increment_epoch(epoch):
    new_epoch = epoch.copy()
    new_epoch['epoch'] = epoch['epoch'] + 1
    return new_epoch


def time(epoch):
    new_epoch = epoch.copy()
    new_epoch['walltime'] = datetime.now()
    new_epoch['elased_time'] = new_epoch.walltime - epoch.walltime
    return new_epoch


def iterate(f, x):
    """
    Joel Grus: Learning Data Science Using Functional Python
    """
    return accumulate(repeat(x), lambda fx, _: f(fx))


def repeat_generator(n_repeat, generator):
    while True:
        out = next(generator)
        for _ in range(n_repeat):
            yield out


def append(name, values, dataframe):
    df = dataframe.copy()
    df[name] = values
    return df
