import pandas as pd

def multiagent_get_experiance(state,
                              observations,
                              transition,
                              policies):

    actions = []
    pss = []
    for policy, observation in zip(policies, observations):
        action, ps = policy(observation)
        actions.append(action)
        pss.append(ps)

    state_prime, observation_primes, rewards, terminals, stats = transition(state, actions)

    exps = []
    for observation, action, observation_prime, reward, terminal, policy_stats in zip(observations, actions, observation_primes, rewards, terminals, pss):
        experiance_df = {'state': state,
                         'observation': observation,
                         'action': action,
                         'reward': reward,
                         'state_prime': state_prime,
                         'observation_prime': observation_prime,
                         'terminal': terminal,
                         'policy_stats': policy_stats}

        experiance_df.update(stats)
        experiance_df.update(policy_stats)

        exps.append(experiance_df)

    return exps


def multiagent_experiance_generator(initial_states,
                                    transition,
                                    policies):
    try:
        initial_state, initial_observations = next(initial_states)
        exps = multiagent_get_experiance(initial_state,
                                         initial_observations,
                                         transition,
                                         policies)
        yield exps
        while True:
            if exps[0]['terminal']:
                initial_state, initial_observations = next(initial_states)
                exps = multiagent_get_experiance(initial_state,
                                                 initial_observations,
                                                 transition,
                                                 policies)
                yield exps
            else:
                observations = [e['observation_prime'] for e in exps]
                exps = multiagent_get_experiance(exps[0]['state_prime'],
                                                 observations,
                                                 transition,
                                                 policies)
                yield exps
    except StopIteration:
        return 

def multiagent_variable_opponent_experiance_generator(initial_states,
                                                      transition,
                                                      policy,
                                                      opponent_policy_generator):
    try:
        initial_state, initial_observations = next(initial_states)
        policies = [policy, next(opponent_policy_generator)]
        exps = multiagent_get_experiance(initial_state,
                                         initial_observations,
                                         transition,
                                         policies)
        yield exps
        while True:
            if exps[0]['terminal']:
                policies = [policy, next(opponent_policy_generator)]
                initial_state, initial_observations = next(initial_states)
                exps = multiagent_get_experiance(initial_state,
                                                 initial_observations,
                                                 transition,
                                                 policies)
                yield exps
            else:
                observations = [e['observation_prime'] for e in exps]
                exps = multiagent_get_experiance(exps[0]['state_prime'],
                                                 observations,
                                                 transition,
                                                 policies)
                yield exps
    except StopIteration:
        return 
