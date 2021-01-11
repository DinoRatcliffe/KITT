import numpy as np
import pandas as pd

from kitt.experiance_generators import episodes_generator
from kitt.utils import take

def evaluate_policy(games, initial_states, transition, pi, epoch, key="environment_evaluation"):
    data = list(map(lambda e: [e.score.iloc[-1], len(e.score), sum(e.reward)],
                    take(games, episodes_generator(initial_states,
                                                   transition,
                                                   pi))))
    data = np.array(data)
    epoch[key] = pd.Series([data[:, 0],
                            data[:, 1],
                            data[:, 2]], index=[key + "-scores",
                                                key + "-lengths",
                                                key + "-reward-sum"])
    return epoch

def evaluate_multiagent_policy(games, initial_states, transition, pi, epoch, key="environment_evaluation"):
    data = list(map(lambda e: [e.score.iloc[-1], len(e.score), sum(e.reward), 1 if e.score.iloc[-1] > 0 else 0],
                    take(games, episodes_generator(initial_states,
                                                   transition,
                                                   pi))))
    data = np.array(data)
    epoch[key] = pd.Series([data[:, 0],
                            data[:, 1],
                            data[:, 2],
                            data[:, 3]], index=[key + "-scores",
                                                key + "-lengths",
                                                key + "-reward-sum",
                                                key + "-wins"])
    return epoch
