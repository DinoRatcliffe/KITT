#########
# UTILS #
#########
from collections import deque, namedtuple
ContextObservation = namedtuple('ContextObservation', 'observation, context')

def wrap_context(context_size, observation_fn, transition_fn): 
    context = deque(maxlen=context_size)

    def transition_context(state, action):
        state_prime, observation_prime, reward, done, data = transition_fn(state, action)
        context.append((observation_fn(state), action))
        context_observation = ContextObservation(observation_prime, context.copy())

        if done:
            context.clear()

        return state_prime, context_observation, reward, done, data

    return transition_context
