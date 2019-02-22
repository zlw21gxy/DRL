import tensorflow as tf
import numpy as np
from env import CarlaEnv
try:
    from carla.client import CarlaClient
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, \
        TURN_RIGHT, TURN_LEFT, LANE_FOLLOW
except Exception as e:
    print("Failed to import Carla python libs, try setting $CARLA_PY_PATH")
    raise e

env = CarlaEnv()
obs = env.reset()
act = (1, 0)
done = False

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 3,
    "TURN_RIGHT": 6,
    "TURN_LEFT": 5,
    "LANE_FOLLOW": 3,
}


DISCRETE_ACTIONS = {
    # coast
    0: [0.0, 0.0],
    # turn left
    1: [0.0, -0.5],
    # turn right
    2: [0.0, 0.5],
    # forward
    3: [1.0, 0.0],
    # brake
    4: [-0.5, 0.0],
    # forward left
    5: [1.0, -0.5],
    # forward right
    6: [1.0, 0.5],
    # brake left
    7: [-0.5, -0.5],
    # brake right
    8: [-0.5, 0.5],
}

num = 3
t = 0
while not done:
    act = DISCRETE_ACTIONS[num]
    obs, rew, done, info = env.step(0.5*np.array(act))
    num = COMMAND_ORDINAL[info['next_command']]
    t += 1
