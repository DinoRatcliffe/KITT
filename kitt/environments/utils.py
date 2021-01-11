#########
# UTILS #
#########
from collections import deque, namedtuple

def frame_stack(stack_size, transition):
    context = deque(maxlen=stack_size)

    def transition_stacked(state, action):
        state_prime, observation_prime, reward, done, stats = transition(state, action)

        if len(context) < stack_size:
            for _ in range(stack_size):
                context.append(np.zeros_like(observation_prime))

        context.append(observation_prime)

        if done:
            context.clear()

        return state_prime, np.concatenate(context, len(observation_prime.shape)-1), reward, done, data
