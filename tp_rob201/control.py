""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar, counter):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1
    # default speed
    speed = 0.1

    laser_dist = lidar.get_sensor_values()

    is_close = laser_dist[180] < 60 or laser_dist[150] < 60 or laser_dist[30] < 60

    # see if the front part of the car is close to the wall (or was already close)
    if is_close or counter > 0:
        speed = 0
        # randomly choses the side to turn
        rotation_speed = random.randint(0, 1)            
    else:
        rotation_speed = 0

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    command = {"forward": 0,
               "rotation": 0}

    return command
