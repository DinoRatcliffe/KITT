import pandas as pd
import numpy as np

from kitt.parameters import constant
from kitt.policies import stochastic_policy
from kitt.experiance_generators import get_experiance
from kitt.planning import ContextObservation

##################
# Hyperparamters #
##################
def rhea_hyperparameters(rollout_length=constant(50),
                         mutation_probability=constant(0.25),
                         number_evaluations=constant(100),
                         use_shift_buffer=constant(True),
                         flip_at_least_once=constant(False)):
    """RHEA Hyperparameters

    TODO (ratcliffe@dino.ai): give explanation of each parameter
    """
    while True:
        yield pd.Series([next(rollout_length),
                         next(mutation_probability),
                         next(number_evaluations),
                         next(use_shift_buffer),
                         next(flip_at_least_once)],
                         index=['rollout_length',
                                'mutation_probability',
                                'number_evaluations',
                                'use_shift_buffer',
                                'flip_at_least_once'])

#########
# UTILS #
#########
def shiftbuffer_rhea(rhea):
    last_best = [None]
    def shift_rhea(state):
        logits, data = rhea(state, previous_best_solution=last_best[0])
        last_best[0] = data['best_solution']
        return logits, data

    return shift_rhea


#############
# Functions #
#############
def shift_solution_random(solution,
                          number_actions):
    new_solution = solution[1:]
    new_solution.append(np.random.randint(number_actions))
    return new_solution


def random_solution(rollout_length,
                    number_actions):
    return [np.random.randint(number_actions) for _ in range(rollout_length)]


def random_mutations(solution,
                     number_actions,
                     mutation_probability,
                     number_evaluations,
                     flip_at_least_once):
    candidate_solutions = []
    for _ in range(number_evaluations):
        new_solution = solution.copy()

        if flip_at_least_once:
            random_action_idx = np.random.randint(len(solution))
            new_solution[random_action_idx] = np.random.randint(number_actions)

        for index, mutate_value in enumerate(np.random.random(len(new_solution))):
            if mutate_value < mutation_probability:
                new_solution[index] = np.random.randint(number_actions)

        candidate_solutions.append(new_solution)

    return candidate_solutions

def guided_mutations(state,
                     solution,
                     number_actions,
                     model,
                     observation_function,
                     action_policy,
                     number_evaluations):
    candidate_solutions = []
    for _ in range(number_evaluations):
        new_solution = solution.copy()
        p_mutation, action_logits = model(observation_function(state), solution)

        # we only passed a single state and solution, so only care about
        # the first result
        p_mutation = p_mutation.numpy()[0]
        action_logits = action_logits.numpy()[0]

        for i, p, action_logit, random_value in zip(range(len(p_mutation)),
                                                    p_mutation,
                                                    action_logits,
                                                    np.random.random(len(p_mutation))):
            if random_value < p:
                new_solution[i] = action_policy(action_logit)[0]
        candidate_solutions.append(new_solution)

    return candidate_solutions

def guided_solution(model,
                    observation_function,
                    action_policy,
                    rollout_length,
                    number_actions,
                    state):
    solution = []
    p_mutation, action_logits = model(observation_function(state), [0] * rollout_length)
    for logit in action_logits:
        solution.append(action_policy(logit)[0])
    return solution 

def guided_iteration_solution(model,
                              observation_function,
                              action_policy,
                              rollout_length,
                              transition_fn,
                              state):
    current_state = state
    solution = []
    for i in range(rollout_length):
        p_mutation, action_logits = model([observation_function(current_state)], [0])
        action = action_policy(action_logits.numpy()[0])[0]
        current_state, _, _, _, _ = transition_fn(current_state, action)
        solution.append(action)

    return solution

def guided_mutations_iteration(transition_fn,
                               state,
                               solution,
                               number_actions,
                               model,
                               observation_function,
                               action_policy,
                               number_evaluations):
    candidate_solutions = []
    current_states = [state] * number_evaluations
    for i, random_values in zip(range(len(solution)),
                                np.random.random([len(solution), number_evaluations])):
        current_actions = [solution[i]] * number_evaluations
        current_obs = [observation_function(state) for state in current_states]
        p_mutations, action_logits = model(current_obs, current_actions)

        new_actions = []

        for p_mut, action_log, cur_ac, rand in zip(p_mutations.numpy(),
                                                   action_logits.numpy(),
                                                   current_actions,
                                                   random_values):
            new_ac = mutate_action(p_mut, rand, cur_ac, action_log, action_policy)
            new_actions.append(new_ac)

#        current_states = list(map(lambda x: x[0], [transition_fn(state, act) for state, act in zip(current_states, current_actions)]))
        current_states = transition_fn(current_states, current_actions)
        candidate_solutions.append(new_actions)

    return np.transpose(candidate_solutions)


def mutate_action(p_mutation, random_value, current_action, action_logits, action_policy):
    if random_value < p_mutation:
        current_action = action_policy(action_logits)[0]
    return current_action

def calculate_fitness(fitness_fn,
                      state,
                      model_transition,
                      candidate_solutions):
    fitnesses = []
    for solution in candidate_solutions:
        experiances = []
        current_state = state
        for action in solution:
            experiance = get_experiance(current_state,
                                        None,
                                        model_transition,
                                        lambda x: (action, pd.Series()))
            current_state = experiance.state_prime
            experiances.append(experiance)
            if experiance.terminal:
                break;
        fitnesses.append(fitness_fn(experiances))
    return fitnesses

def reward_sum_fitness(experiances):
    return sum([exp['reward'] for exp in experiances])

def final_score_fitness(experiances):
    return experiances[-1]['score']

def sequential_fitness(fitness_fn,
                       model_transition,
                       state,
                       candidate_solutions,
                       resamples=10):
    fitnesses = []
    states = []
    for solution in candidate_solutions:
        solution_fitnesses = []
        for _ in range(resamples):
            solution_states = []
            experiances = []
            current_state = state
            for action in solution:
                experiance = get_experiance(current_state,
                                            None,
                                            model_transition,
                                            lambda x: (action, {}))
                current_state = experiance['state_prime']
                experiances.append(experiance)
                solution_states.append(current_state)
                if experiance['terminal']:
                    break;
            solution_fitnesses.append(fitness_fn(experiances))
        fitnesses.append(np.mean(solution_fitnesses))
        states.append(solution_states)

    return fitnesses, states

def remote_fitness(model_transition,
                   state,
                   candidate_solutions):
    return model_transition(state, 0, candidate_solutions)


#############
# Algorithm #
#############
def rhea_model(hyperparameters,
               number_actions,
               transition_fn,
               fitness_fn,
               state,
               guidance_model=None,
               guidance_iteration=False,
               guidance_observation_function=None,
               guidance_action_policy=stochastic_policy,
               previous_best_solution=None):

        context = None
        if isinstance(state, ContextObservation):
            state, context = state

        # create current base solution
        params = next(hyperparameters)
        if params.use_shift_buffer and previous_best_solution is not None:
            base_solution = shift_solution_random(previous_best_solution, 
                                                  number_actions)
        else:
            if guidance_model is None:
                base_solution = random_solution(params.rollout_length,
                                                number_actions)
            elif guidance_iteration:
                base_solution = guided_iteration_solution(guidance_model,
                                                          guidance_observation_function,
                                                          guidance_action_policy,
                                                          params.rollout_length,
                                                          transition_fn,
                                                          state)
            else:
                base_solution = guided_solution(guidance_model,
                                                guidance_observation_function,
                                                guidance_action_policy,
                                                params.rollout_length,
                                                number_actions,
                                                state)
 
        # mutate solutions
        if guidance_model is None:
            candidate_solutions = random_mutations(base_solution,
                                                   number_actions,
                                                   params.mutation_probability,
                                                   params.number_evaluations,
                                                   params.flip_at_least_once)
        else:
            if guidance_iteration:
                candidate_solutions = guided_mutations_iteration(transition_fn,
                                                                 state,
                                                                 base_solution,
                                                                 number_actions,
                                                                 guidance_model,
                                                                 guidance_observation_function,
                                                                 guidance_action_policy,
                                                                 params.number_evaluations)
            else:
                candidate_solutions = guided_mutations(state,
                                                       base_solution,
                                                       number_actions,
                                                       guidance_model,
                                                       guidance_observation_function,
                                                       guidance_action_policy,
                                                       params.number_evaluations)

        # evaluate fitness of solutions
        if context is not None:
            fitnesses, states = fitness_fn(transition_fn, state, candidate_solutions, context=context)
        else:
            fitnesses, states = fitness_fn(transition_fn, state, candidate_solutions)

        # set one hot action based on best_solution
        best_solution = candidate_solutions[np.argmax(fitnesses)]
        logits = [0] * number_actions
        logits[best_solution[0]] = 1

        return logits, {'best_solution': best_solution,
                        'solutions': candidate_solutions,
                        'solution_states': states,
                        'fitnesses': fitnesses}


#############################################
# TODO (ratcliffe@dino.ai): Should be moved #
#############################################
import sonnet as snt
import tensorflow as tf
from functools import partial
from kitpy.utils import one_hot
class RHEAGuidanceModel(snt.Module):
    def __init__(self, mutation_model, action_model, name="RHEAGuidanceModel"):
        super(RHEAGuidanceModel, self).__init__(name=name)
        self._mutation_model = mutation_model
        self._action_model = action_model

    @property
    def trainable_variables(self):
        return list(map(lambda x: x.deref(), set(map(lambda x: x.experimental_ref(), self._mutation_model.trainable_variables + self._action_model.trainable_variables))))

    @property
    def variables(self):
        return list(map(lambda x: x.deref(), set(map(lambda x: x.experimental_ref(), self._mutation_model.variables + self._action_model.variables))))

    def __call__(self, state, actions):
        # turn actions into onehot
        one_hot_actions = tf.one_hot(actions, 2)

        # combine with state, observation really but planning wants state we want observation
        #network_input = tf.concat(tf.constant(state[0]), tf.reshape(tf.constant(one_hot_actions, tf.float64), [-1]), 0)
        state_input = state
        action_input = tf.reshape(tf.cast(one_hot_actions, tf.float64), [-1])
        net_input = tf.expand_dims(tf.concat([state_input, action_input], 0), 0)

        # pass through mutation model to get mutation probabilities
        p_mutation = self._mutation_model(net_input)

        # pass throught action_model to get action logits
        action_logits = self._action_model(net_input)

        # return mutation probs and action logits
        return p_mutation, action_logits

class RHEAIterationGuidanceModel(snt.Module):
    def __init__(self, mutation_model, action_model, name="RHEAGuidanceModel"):
        super(RHEAIterationGuidanceModel, self).__init__(name=name)
        self._mutation_model = mutation_model
        self._action_model = action_model

    @property
    def trainable_variables(self):
        return list(map(lambda x: x.deref(), set(map(lambda x: x.experimental_ref(), self._mutation_model.trainable_variables + self._action_model.trainable_variables))))

    @property
    def variables(self):
        return list(map(lambda x: x.deref(), set(map(lambda x: x.experimental_ref(), self._mutation_model.variables + self._action_model.variables))))

    def __call__(self, states, actions):
        # turn actions into onehot
        one_hot_actions = tf.cast(tf.one_hot(actions, 6*5), tf.float64)

        #net_input = tf.concat((states, tf.cast(one_hot_actions, tf.float64)), 1)

        # pass through mutation model to get mutation probabilities
        #p_mutation = self._mutation_model(net_input)
        #p_mutation = self._mutation_model([tf.constant(states), one_hot_actions])
        p_mutation = tf.squeeze(self._mutation_model([tf.constant(states), one_hot_actions]))

        # pass throught action_model to get action logits
        action_logits = self._action_model(tf.constant(states))

        # return mutation probs and action logits
        return p_mutation, action_logits


def planning_policy(planning_approach,
                    policy,
                    policy_params):
    p_params  = list(map(lambda value: next(value), policy_params.values))
    # TODO (ratcliffe@dino.ai): sort this out as it will slow down training etc.
    epoch_params = {} #pd.Series(p_params, index=policy_params.index)

    while True:
        def out_policy(state):
            logits, stats = planning_approach(state)
            return policy(*p_params, logits), stats

        yield out_policy, epoch_params
