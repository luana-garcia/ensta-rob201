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

def calculate_grad_attr(current_pose, goal_pose, K_goal = 1):
    d_min = np.linalg.norm(goal_pose[:2] - current_pose[:2])
    grad_attr = (K_goal / d_min) * (goal_pose[:2] - current_pose[:2])
    return grad_attr

def compute_grad_poten_repul(current_pose, goal_pose, K_goal = 1):
    d_min = np.linalg.norm(goal_pose[:2] - current_pose[:2])
    grad_attr = (K_goal / d_min) * (goal_pose[:2] - current_pose[:2])
    return grad_attr

def compute_repulsive_gradient(lidar, current_pose, d_safe=60, K_obs=800):
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    valid = (distances > 5.0) & (distances < 300.0)
    distances = distances[valid]
    angles = angles[valid]

    grad_rep = np.zeros(2)
    x, y, theta = current_pose

    for d, a in zip(distances, angles):
        if d < d_safe:
            obs_x = x + d * np.cos(theta + a)
            obs_y = y + d * np.sin(theta + a)
            q_obs = np.array([obs_x, obs_y])
            q = np.array([x, y])
            delta = q - q_obs
            dist = np.linalg.norm(delta)

            if dist > 1e-3:
                repulsion = K_obs * (1.0/dist - 1.0/d_safe) * (1.0 / dist**3) * delta
                grad_rep += repulsion

    return grad_rep

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
    speed = 0.1
    speed_rot = 0

    grad_attr = calculate_grad_attr(current_pose, goal_pose)
    grad_repuls = compute_repulsive_gradient(lidar, current_pose)
    grad_total = grad_attr + grad_repuls

    if(grad_attr[0] < 0):
        speed = 0
        speed_rot = 0
    else:
        angle_to_goal = np.arctan2(grad_total[1], grad_total[0])
        angle_diff = angle_to_goal - current_pose[2]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        speed_rot = np.clip(2.0 * angle_diff, -1.0, 1.0)

        print(f'{grad_total[0]:.2f} {grad_total[1]:.2f}')

    command = {"forward": speed,
               "rotation": speed_rot}

    return command
