from typing import TypeVar, Callable, Tuple, Union, List, Generator
import datetime

import numpy as np
import pandas as pd

#################
# Generic Types #
#################
T = TypeVar('T')
PredicateType = Callable[[T], bool]


##############
# Base Types #
##############
StateType = TypeVar('StateType')
ObservationType = TypeVar('ObservationType')
ModelType = TypeVar('ModelType')
ApproachStateType = TypeVar('ApproachStateType')
ActionType = Union[int, List[float]]
ActionDiscreet = int
ActionContinuous = List[float]
PolicyParamsType = TypeVar('PolicyParamsType') # TODO (dr@dino.ai): should be dataframe
EnvDataType = TypeVar('EnvDataType') # TODO (dr@dino.ai): should be dataframe


#####################
# Observation Types #
#####################
# @comment nptyping doesn't really work but it is better than nothing
# Image Observations #
ImgGreyScaleType = np.ndarray
ImgRGBType = np.ndarray
ImgRGBAType = np.ndarray


##################
# Function Types #
##################
TransitionType = Callable[[StateType, int], Tuple[StateType, ObservationType, float, bool, EnvDataType]]
                 #Callable[[StateType, int], Tuple[StateType, ObservationType, float, bool, EnvDataType]]


##################
# Model Policies #
##################
ModelPolicyDiscreet = Callable[[StateType], ActionDiscreet]
ModelPolicyContinuous = Callable[[StateType], ActionContinuous]
ModelPolicyType = Union[ModelPolicyDiscreet, ModelPolicyContinuous]


###########################
# Model Policy Generators #
###########################
ModelPolicyGenerator = Generator[Tuple[ModelPolicyType, PolicyParamsType],
                                 None,
                                 None]
ModelPolicyDiscreetGenerator = Generator[Tuple[ModelPolicyDiscreet, PolicyParamsType],
                                         None,
                                         None]
ModelPolicyContinuousGenerator = Generator[Tuple[ModelPolicyContinuous, PolicyParamsType],
                                           None,
                                           None]

PolicyType = Callable[[T], ActionType]
InitialStateGeneratorType = Generator[Tuple[StateType, ObservationType], None, None]
