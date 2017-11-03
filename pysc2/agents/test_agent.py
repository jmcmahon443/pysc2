# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy
import time
import sonnet as snt
import tensorflow as tf

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

## HARDCODED ZERG
# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_OVERLOAD = actions.FUNCTIONS.Train_Overlord_quick.id
_TRAIN_DRONE = actions.FUNCTIONS.Train_Drone_quick.id
_BUILD_SPAWNING_POOL = actions.FUNCTIONS.Build_SpawningPool_screen.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit ID
_ZERG_LARVA = 151
_ZERG_EGG = 103
_ZERG_DRONE = 104
_ZERG_DRONE_BURROWED = 116
_ZERG_OVERLORD = 106
_ZERG_QUEEN = 126
_ZERG_QUEEN_BURROWED = 125
_ZERG_ZERGLING = 105
_ZERG_ZERGLING_BURROWED = 119
_ZERG_BANELING_COCOON = 9
_ZERG_BANELING = 9
_ZERG_HATCHERY = 86
_ZERG_SPAWNING_POOL = 89

# Parameters
_PLAYER_SELF = 1
_SCREEN = [0]

class TestAgent(base_agent.BaseAgent):
  """Joe's test agent for starcraft."""
  base_top_left = None
  larva_selected = False
  overload_trained = False
  spawning_pool_built = False
  drone_selected = False

  def transformLocation(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
      return [x - x_distance, y - y_distance]
    return [x + x_distance, y + y_distance]

  def __init__(self):
    super(TestAgent, self).__init__()
    self.mineral_reward = 0

  def setup(self, obs_spec, action_spec):
    super(TestAgent, self).setup(obs_spec, action_spec)
    print("Python version: ")
    print(sys.version)
    print("Python location: ")
    print(os.path.dirname(sys.executable))

  def reset(self):
    super(TestAgent, self).reset()

  def step(self, obs):
    super(TestAgent, self).step(obs)

    ## RANDOM ACTIONS
    # function_id = numpy.random.choice(obs.observation["available_actions"])
    # args = [[numpy.random.randint(0, size) for size in arg.sizes]
    #         for arg in self.action_spec.functions[function_id].args]
    # return actions.FunctionCall(function_id, args)

    ## STEP AND PRINT OBESERVATION SPACE
    # print("\n")
    # print("Collected minerals: ")
    # print(obs.observation['score_cumulative'][7])
    # print("Collection rate minerals: ")
    # print(obs.observation['score_cumulative'][9])
    # input("Press ENTER key to continue.")

    ## TODO: Store mineral rewards from previous episodes, update a Q-cost matrix.
    # self.mineral_reward+=obs.score.score_details.collected_minerals

    ## TEST GROUND
    # time.sleep(0.5)
    input("Press ENTER key to continue.")

    if self.base_top_left is None:
      player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
      self.base_top_left = player_y.mean() <= 31

    if not self.overload_trained:
      if not self.larva_selected:
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _ZERG_LARVA).nonzero()
        print(unit_x)
        print(unit_y)
        target = [unit_x[0], unit_y[0]]
        self.larva_selected = True
        return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
      elif _TRAIN_OVERLOAD in obs.observation["available_actions"]:
        self.overload_trained = True
        self.larva_selected = False
        return actions.FunctionCall(_TRAIN_OVERLOAD, [_SCREEN])

    if self.overload_trained and not self.spawning_pool_built:
      if not self.drone_selected:
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _ZERG_DRONE).nonzero()
        target = [unit_x[0], unit_y[0]]
        self.drone_selected = True
        return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
      elif _BUILD_SPAWNING_POOL in obs.observation["available_actions"]:
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _ZERG_HATCHERY).nonzero()
        target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
        self.spawning_pool_built = True
        self.drone_selected = False
        return actions.FunctionCall(_BUILD_SPAWNING_POOL, [_SCREEN, target])

    ## TAKE NO ACTION
    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
    # return actions.FunctionCall(0, [])