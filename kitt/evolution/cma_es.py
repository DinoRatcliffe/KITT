import cma
import pandas as pd
import numpy as np

from kitt.utils import take
from kitt.experiance_generators import episodes_generator
from kitt.parameters import constant
from kitt.models import sonnet_set_parameters, sonnet_get_parameters

####################
# Hyper Parameters #
####################
def cma_es_hyperparams(sigma_init=constant(1.0),
                       population_size=constant(20),
                       num_eval_episodes=constant(20),
                       weight_decay=constant(0.01)):
    while True:
        yield pd.Series([next(sigma_init),
                         next(population_size),
                         next(num_eval_episodes),
                         next(weight_decay)],
                        index=['sigma_init',
                               'population_size',
                               'num_eval_episodes',
                               'weight_decay'])

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



############
# Approach #
############
def cma_es(hyperparameters,
           env_fn,
           model_fn,
           epoch):
    # 1. inital setup
    epoch = epoch.copy()
    epoch['cma_state'] = epoch['cma_state'].copy()
    hyperparams = next(hyperparameters)
    model = epoch.cma_state.model

    # increment curriculum if told to
    if 'increment_curriculum' in epoch['cma_state'] and epoch['cma_state']['increment_curriculum']:
        epoch.cma_state.workers.increment_curriculum()
        epoch['cma_state']['increment_curriculum'] = False

    if 'cma_controller' not in epoch.cma_state:
        initial_params = sonnet_get_parameters(model)
        epoch.cma_state['cma_controller'] = cma.CMAEvolutionStrategy(initial_params,
                                                                     hyperparams.sigma_init,
                                                                     {'popsize': hyperparams.population_size})
    if 'workers' not in epoch.cma_state:
        epoch.cma_state['workers'] = WorkerController(env_fn,
                                                      model_fn,
                                                      hyperparams.num_eval_episodes,
                                                      size=6)
        
    workers = epoch.cma_state.workers
    controller = epoch.cma_state.cma_controller

    # 2. Get next generation of parameters
    solutions = np.array(controller.ask())

    # 3. Calculate fitness of solutions
    fitnesses = workers.calculate_fitness(solutions, hyperparams.num_eval_episodes)

    # 4. inform controller of fitness for each solution
    controller.tell(solutions, [-fitness for fitness in fitnesses])

    # 5. Return best solution
    result = controller.result
    epoch.cma_state.model = sonnet_set_parameters(epoch.cma_state.model,
                                                  controller.result[0])

    epoch['cma_state']['best_solution'] = controller.result[0]
    epoch['cma_state']['log_data'] = pd.Series([np.array(fitnesses)], index=['fitnesses'])
    
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
    initial_states, transition, increment_curriculum = pickle.loads(env_fn_wrapper)()
    model, policy_generator = pickle.loads(model_fn_wrapper)()

    # must pass one state through model before we create trainable variables
    state, obs = next(initial_states)
    next(policy_generator)[0](obs)

    try:
        while True:
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
