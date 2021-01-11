import pandas as pd

from kitt.experiance_generators import experiance_generator, multiagent_experiance_generator


def multiprocess_generator(env_fn, policy, n_processes):
    controller = WorkerController(env_fn, n_processes)

    while True:
        # get observations to pass throught policy
        observations = controller.get_observations()

        # get actions
        actions = policy(observations)

        # give actions to controller
        experiances = controller.transition(actions[0])
        yield experiances

def multiprocess_multiagent_generator(env_fn, policies, n_processes):
    controller = WorkerController(env_fn, n_processes, worker=multiagent_worker)

    while True:
        for policy in policies:
            # get observations to pass throught policy
            observations = controller.get_observations()

            # get actions
            actions = policy(observations)
            controller.send_actions(actions)

        # give actions to controller
        experiances = controller.get_experiances()
        yield experiances


####################
# Multi-Processing #
####################
import multiprocessing as mp
import os
import cloudpickle
import pickle

def multiagent_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()

    initial_states, transition = pickle.loads(env_fn_wrapper)()

    def remote_blocking_policy(observation):
        try:
            remote.send(observation)
            action = remote.recv()
        except KeyboardInterrupt:
            print('Remote Genrator: got KeyboardInterrupt')

        return action, {}

    exp_gen = multiagent_experiance_generator(initial_states, transition, [remote_blocking_policy, remote_blocking_policy])

    while True:
        try:
            remote.send(next(exp_gen))
        except KeyboardInterrupt:
            print('Remote Genrator: got KeyboardInterrupt')

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()

    initial_states, transition = pickle.loads(env_fn_wrapper)()

    def remote_blocking_policy(observation):
        try:
            remote.send(observation)
            action = remote.recv()
        except KeyboardInterrupt:
            print('Remote Genrator: got KeyboardInterrupt')

        return action, {}

    exp_gen = experiance_generator(initial_states, transition, remote_blocking_policy)
    while True:
        try:
            remote.send(next(exp_gen))
        except KeyboardInterrupt:
            print('Remote Genrator: got KeyboardInterrupt')


class WorkerController():
    def __init__(self, env_fn, size, worker=worker):
        self.waiting = False
        self.closed = False

        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(size)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, cloudpickle.dumps(env_fn))) for (work_remote, remote) in zip(self.work_remotes, self.remotes)]

        for p in self.ps:
            p.daemon = True
            with clear_mpi_env_vars():
                p.start()

        for remote in self.work_remotes:
            remote.close()

    def send_actions(self, actions):
        for action, remote in zip(actions, self.remotes):
            remote.send(action)

    def get_experiances(self):
        experiances = []
        for remote in self.remotes:
            experiances.append(remote.recv())
        return experiances

    def get_observations(self):
        obs = []
        for remote in self.remotes:
            obs.append(remote.recv())

        return obs

    def transition(self, actions):
        self.send_actions(actions)
        return self.get_experiances()


import contextlib
@contextlib.contextmanager
def clear_mpi_env_vars():
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)
