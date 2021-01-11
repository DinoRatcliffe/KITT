"""Set of traditional control task environments."""
import math 

from typing import Tuple
import cv2

from gym.envs.classic_control.cartpole import CartPoleEnv
import numpy as np

from kitt.types import (InitialStateGeneratorType,
                         TransitionType)


#########################
# Observation Functions #
#########################
def cartpole_image_observation(width, height):
    screen_width = width 
    screen_height = height

    world_width = 2.4*2
    scale = screen_width/world_width
    carty = screen_height/2 # TOP OF CART
    cartwidth = screen_width / 5
    cartheight = screen_height / 5

    def render(state):
        image = np.zeros((screen_height, int(screen_width), 1))

        cv2.line(image,
                 (0, int(carty)),
                 (screen_width, int(carty)),
                 255)

        polelen = scale * (3.5 * state[3][2])

        cartx = state[0][0]*scale+screen_width/2.0
        cv2.rectangle(image,
                      (int(cartx-cartwidth/2), int(carty-cartheight/2)),
                      (int(cartx+cartwidth/2), int(carty+cartheight/2)),
                      255,
                      thickness=-1)

        theta = state[0][2]
        cv2.line(image,
                 (int(cartx), int(carty)),
                 (int((np.cos(theta-math.pi/2)*polelen)+cartx), int((np.sin(theta-math.pi/2)*polelen) + carty)),
                 255,
                 thickness=int(scale/2),
                 lineType=4)

        # crop excess image
        return image

    return render


def cartpole_internal_observation(state):
    return state[0]


def cartpole_internal_dynamics_observation(state):
    return np.concatenate((state[0], state[3]))

def cartpole_internal_dynamics_observation_single(parameter):

    def obs_fn(state):
        extra_data = []
        if parameter == 'masscart':
            extra_data.append(state[3][0])
        elif parameter == 'masspole':
            extra_data.append(state[3][1])
        elif parameter == 'length':
            extra_data.append(state[3][2])
        elif parameter == 'force_mag':
            extra_data.append(state[3][3])

        return np.concatenate((state[0], extra_data))

    return obs_fn

#############################
# Cartpole Reward Functions #
#############################
def pole_angle_reward(angle=12):
    def reward_fn(state, aciton, state_prime):
        theta_thresh = angle * 2 * np.pi / 360
        reward = 1 if -theta_thresh <= state_prime[0][2] <= theta_thresh else -1
        return reward
    return reward_fn


def mse_target_state_reward(target_state, observation_fn=cartpole_internal_observation):
    def reward_fn(state, action, state_prime):
        return -np.mean((observation_fn(target_state) - observation_fn(state_prime)) ** 2)
    return reward_fn


def joint_reward(reward_fns, weights=[]):
    def reward_fn(state, action, state_prime):
        rewards = [fn(state, action, state_prime) for fn in reward_fns]
        rewards = np.array(rewards)
        if len(weights) > 0:
            rewards *= weights
        return sum(rewards)
    return reward_fn

######################
# Terminal Functions #
######################
def pole_angle_terminal(angle=12):
    def terminal_fn(state, action, state_prime):
        theta_thresh = angle * 2 * np.pi / 360
        return state_prime[0][2] <= -theta_thresh or state_prime[0][2] >= theta_thresh
    return terminal_fn


def cart_x_terminal(xlimit=2.4):
    def terminal_fn(state, action, state_prime):
        return state_prime[0][0] < -xlimit or state_prime[0][0] > xlimit
    return terminal_fn


def max_length_terminal(length=200):
    def terminal_fn(state, action, state_prime):
        return state_prime[2] >= 200
    return terminal_fn


def joint_terminal(terminal_fns):
    def terminal_fn(state, action, state_prime):
        terminal = False
        for term_fn in terminal_fns:
            terminal = terminal or term_fn(state, action, state_prime)
        return terminal
    return terminal_fn



##################
# Main Functions # 
##################
def cartpole_initial_state_sequence(masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, observation_function=cartpole_internal_observation, reward_function=pole_angle_reward(), terminal_function=joint_terminal([pole_angle_terminal(), cart_x_terminal(), max_length_terminal()])) -> InitialStateGeneratorType:
    """Initial state, observation generator.

    Geneartor that samples from the set of initial games states, also
    returns the observation for the initial state.
    """
    env = CartPole(masscart=masscart, masspole=masspole, length=length, force_mag=force_mag, observation_function=observation_function, reward_function=reward_function, terminal_function=terminal_function)
    while True:
        env.reset()
        # state and observation are the same for vanila cartpole
        yield env.state, env.observation


def cartpole(masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, unlimited_theta=False, observation_function=cartpole_internal_observation, reward_function=pole_angle_reward(), terminal_function=joint_terminal([pole_angle_terminal(), cart_x_terminal(), max_length_terminal()])) -> Tuple[InitialStateGeneratorType, TransitionType]:
    """Helper function that creates the cartpole env

    Returns:
        inital_state_sequence: A generator that gives random initial state.
        transition: A transition function that defines the cartpole dynamics.
    """
    env = CartPole(masscart=masscart, masspole=masspole, length=length, force_mag=force_mag, unlimited_theta=unlimited_theta, observation_function=observation_function, reward_function=reward_function, terminal_function=terminal_function)
    return cartpole_initial_state_sequence(observation_function=observation_function, masscart=masscart, masspole=masspole, length=length, force_mag=force_mag), env.transition


def cartpole_parameterised(parameter, value, unlimited_theta=False, observation_function=cartpole_internal_observation, reward_function=pole_angle_reward(), terminal_function=joint_terminal([pole_angle_terminal(), cart_x_terminal(), max_length_terminal()])):
    if parameter == 'masscart':
        return cartpole(masscart=value, unlimited_theta=unlimited_theta, observation_function=observation_function, reward_function=reward_function, terminal_function=terminal_function)
    elif parameter == 'masspole':
        return cartpole(masspole=value, unlimited_theta=unlimited_theta, observation_function=observation_function, reward_function=reward_function, terminal_function=terminal_function)
    elif parameter == 'force_mag':
        return cartpole(force_mag=value, unlimited_theta=unlimited_theta, observation_function=observation_function, reward_function=reward_function, terminal_function=terminal_function)
    elif parameter == 'length':
        return cartpole(length=value, unlimited_theta=unlimited_theta, observation_function=observation_function, reward_function=reward_function, terminal_function=terminal_function)


##############
# Main Class #
##############
class CartPole:
    """Functional wrapper for openai gym cartpole environment.

    Attributes:
        _env (gym.Env): Cached openai cartpole gym environment.
        _score (int): Current score of the environment.
    """

    def __init__(self,
                 masscart=1.0,
                 masspole=0.1,
                 length=0.5,
                 force_mag=10.0,
                 unlimited_theta=False,
                 observation_function=cartpole_internal_observation, 
                 reward_function=pole_angle_reward(),
                 terminal_function=joint_terminal([pole_angle_terminal(), cart_x_terminal(), max_length_terminal()])):
        self._env = CartPoleWrapper(masscart=masscart, masspole=masspole, length=length, force_mag=force_mag, unlimited_theta=unlimited_theta)
        self._score = 0
        self._length = 0
        self._unlimited_theta = unlimited_theta
        self._env.reset()
        self._observation_function = observation_function
        self._reward_function = reward_function
        self._terminal_function = terminal_function


    @property
    def observation(self):
        """Return the observation given the current internal state."""
        return self._observation_function(self.state)

    @property
    def state(self):
        """Return the internal state."""
        dynamics = [self._env.masscart, self._env.masspole, self._env.length, self._env.force_mag]
        return (self._env.state, self._score, self._length, dynamics)


    def reset(self):
        """Reset the current state, observation and logs."""
        self._score = 0
        self._env.reset()


    def render(self, state):
        # Set internal state.
        self._env.reset()
        self._env.state = state[0]

        self._env.render()


    def transition(self,
                   state: Tuple[np.ndarray, int],
                   action: int) -> Tuple[np.ndarray,
                                         np.ndarray,
                                         float,
                                         bool,
                                         dict]:
        """Transition function for the cartpole environment

        Returns:
            new_state: The resulting internal state after taking action
            new_observation: The observation produced by new_state
            reward: The reward recieved from taking action in state
            done: Indicates if new_state is a terminal state
            stats: Pandas series that contains environment statistics
        """
        assert action in [0, 1]

        # Set internal state.
        self._env.set_dynamics(state[3])
        dynamics = [self._env.masscart, self._env.masspole, self._env.length, self._env.force_mag]
        self._env.reset()
        self._env.state = state[0]

        # Perform action.
        env_state, reward, done, _ = self._env.step(action)
        self._length = state[2] + 1

        reward = self._reward_function(state, action, (env_state, self._score, self._length, dynamics))
        self._score = state[1] + reward

        done = self._terminal_function(state, action, (env_state, self._score, self._length, dynamics))

        new_state = (self._env.state, self._score, self._length, dynamics)
        new_observation = self._observation_function(new_state)

        return (new_state,
                np.array(new_observation),
                reward,
                done,
                {'score': self._score})


class CartPoleWrapper(CartPoleEnv):
    def __init__(self, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, unlimited_theta=False):
        CartPoleEnv.__init__(self)

        self.set_dynamics([masscart, masspole, length, force_mag])
        self.unlimited_theta = unlimited_theta
        if unlimited_theta:
            self.theta_threshold_radians = 2 * math.pi

    def set_dynamics(self, dynamics):
        self.masscart = dynamics[0]
        self.masspole = dynamics[1]
        self.total_mass = (self.masspole + self.masscart)
        self.length = dynamics[2]
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = dynamics[3]
