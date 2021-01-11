import pandas as pd
import numpy as np

from kitpy.utils import take
from kitpy.experiance_generators import episodes_generator
from kitpy.parameters import constant
from kitpy.models import sonnet_set_parameters, sonnet_get_parameters


####################
# Hyper Parameters #
####################
def naive_es_hyperparams(population_size=constant(10),
                         minimum_value=constant(0),
                         maximum_value=constant(10),
                         num_eval_episodes=constant(3)):
    while True:
        yield pd.Series([next(population_size),
                         next(minimum_value),
                         next(maximum_value),
                         next(num_eval_episodes)],
                        index=['population_size',
                               'minimum_value',
                               'maximum_value',
                               'num_eval_episodes'])

####################
# Fitness Function #
####################
def fitness(initial_states, transition, pi, num_eval_episodes=16):
    """ mean total reward over n episodes """
    sum_rew = list(take(num_eval_episodes,
                        map(lambda ep: ep.score.iloc[-1],
                            episodes_generator(initial_states, transition, pi))))
    return np.mean(sum_rew)

def rhea_fitness(initial_states, transition, pi, num_eval_episodes=16):
    episode = next(episodes_generator(initial_states, transition, pi))
    return np.mean(list(map(np.mean, episode.fitnesses)))


def breed_individuals(individuals, minimum, maximum):
    return (np.clip(np.mean(individuals) + np.random.normal(0, 1.0), minimum, maximum), 0)


def update_population(population, population_size, minimum=0, maximum=10):
    if len(population) == 1:
        # create initial_population
        new_population = []
        for _ in range(population_size):
            new_population.append((np.random.uniform(minimum, maximum, population[0][0].shape), 0))
    else:
        # mutate current population
        sorted_population = list(reversed(sorted(population, key=lambda x: x[1])))
        print(sorted_population)
        offspring = [breed_individuals([sorted_population[i], sorted_population[i+i]], minimum, maximum) for i in range(0, 4, 2)]
        sorted_population[-len(offspring):] = offspring
        new_population = sorted_population

    return new_population

############
# Approach #
############
def naive_es(hyperparameters,
             env_fn,
             model_fn,
             epoch):

    # 1. inital setup
    epoch = epoch.copy()
    epoch['naive_es_state'] = epoch['naive_es_state'].copy()
    hyperparams = next(hyperparameters)
    model = epoch.naive_es_state.model

    if 'workers' not in epoch.naive_es_state:
        epoch.naive_es_state['workers'] = WorkerController(env_fn,
                                                           model_fn,
                                                           hyperparams.num_eval_episodes,
                                                           size=6)

    if 'population' not in epoch.naive_es_state:
        epoch.naive_es_state.population = ((sonnet_get_parameters(model), 0),)

    workers = epoch.naive_es_state.workers

    # generate next generation
    new_population = update_population(epoch.naive_es_state.population, hyperparams.population_size, hyperparams.minimum_value, hyperparams.maximum_value)

    # calculate fitnesses
    values = [x[0] for x in new_population]
    fitnesses = workers.calculate_fitness(values, hyperparams.num_eval_episodes)
    new_population = list(zip(values, fitnesses))

    # get fittest individual
    fittest_individual = list(sorted(new_population, key=lambda x: x[1]))[-1]
    print(fittest_individual)

    epoch['naive_es_state']['population'] = new_population
    epoch['naive_es_state']['fittest_individual'] = fittest_individual[0]

    # set value
    epoch['naive_es_state']['model'] = sonnet_set_parameters(epoch.naive_es_state.model, np.random.uniform(0, 10, 1))

    return epoch



####################
# Multi-Processing #
####################
import multiprocessing as mp
import os
import cloudpickle
import pickle
def worker(remote, parent_remote, env_fn_wrapper, model_fn_wrapper):
    parent_remote.close()
    initial_states, transition = pickle.loads(env_fn_wrapper)()
    model, policy_generator = pickle.loads(model_fn_wrapper)()

    # must pass one state through model before we create trainable variables
    state, obs = next(initial_states)
    next(policy_generator)[0](obs)

    try:
        closed = False
        while not closed:
            cmd, data = remote.recv()
            if cmd == 'solution':
                solution = data[0]
                num_eval_episodes = data[1]

                model = sonnet_set_parameters(model, solution)
                pi, policy_params = next(policy_generator)
                solution_fitness = fitness(initial_states, transition, pi, num_eval_episodes=num_eval_episodes)
                remote.send(solution_fitness)
            elif cmd == 'increment_curriculum':
                increment_curriculum()
            elif cmd == 'close':
                closed = True
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('CMA Worker: got KeyboardInterrupt')

class WorkerController():
    def __init__(self, env_fn, model_fn, num_episodes, size):
        self.waiting = False
        self.closed = False

        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(size)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, cloudpickle.dumps(env_fn), cloudpickle.dumps(model_fn))) for (work_remote, remote) in zip(self.work_remotes, self.remotes)]

        self._num_episodes = num_episodes

        for p in self.ps:
            p.daemon = True
            with clear_mpi_env_vars():
                p.start()

        for remote in self.work_remotes:
            remote.close()

    def increment_curriculum(self):
        for remote in self.remotes:
            remote.send(('increment_curriculum', {}))


    def calculate_fitness(self, solutions, num_episodes):
        fitnesses = [] 
        solution_idx = 0
        while len(fitnesses) < len(solutions):
            used_remotes = []
            for remote_idx in range(len(self.remotes)):
                if solution_idx < len(solutions):
                    used_remotes.append(remote_idx)
                    self.remotes[remote_idx].send(('solution', (solutions[solution_idx], num_episodes)))
                    solution_idx += 1
                self.waiting = True

            self.waiting = True
            for remote_idx in used_remotes:
                fitnesses.append(self.remotes[remote_idx].recv())
            self.waiting = False
        return fitnesses

    def close(self):
        for remote in self.remotes:
            remote.send(('close', {}))

        for p in self.ps:
            p.join()
            p.close()


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
