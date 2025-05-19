""" Un ensemble de fonctions de contrôle pour la robotique """

import random
import numpy as np


def reactive_obst_avoid(lidar, counter):
    """
    Évitement simple des obstacles
    lidar : objet placebot avec les données du lidar
    """
    # TODO pour TP1
    # vitesse par défaut
    speed = 0.1

    laser_dist = lidar.get_sensor_values()

    is_close = laser_dist[180] < 60 or laser_dist[150] < 60 or laser_dist[30] < 60

    # vérifier si la partie avant du robot est proche du mur (ou l'était déjà)
    if is_close or counter > 0:
        speed = 0
        # choisit aléatoirement le côté vers lequel tourner
        rotation_speed = random.randint(0, 1)            
    else:
        rotation_speed = 0

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command

def calculate_grad_attr(current_pose, goal_pose, K_goal):
    vec_to_goal = goal_pose[:2] - current_pose[:2]
    dist_to_goal = np.linalg.norm(vec_to_goal)
    grad_attr = (K_goal / dist_to_goal) * (vec_to_goal)
    return dist_to_goal, grad_attr

def compute_repulsive_gradient(lidar, current_pose, d_safe, K_obs):
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

def potential_field_control(lidar, current_pose, goal_pose, isStuck=False):
    """
    Contrôle utilisant un champ de potentiel pour atteindre l’objectif et éviter les obstacles
    lidar : objet placebot avec les données du lidar
    current_pose : [x, y, theta] tableau numpy, position actuelle dans le repère odom ou monde
    goal_pose : [x, y, theta] tableau numpy, position cible dans le repère odom ou monde
    Remarques : Comme le lidar et l’odométrie sont des données locales, l’objectif et le gradient
    seront définis soit dans le repère du robot (x,y) (centré sur le robot, x vers l’avant, y à gauche),
    soit dans le repère odométrique (centré/aligne sur la pose initiale, x vers l’avant, y à gauche)
    """
    # TODO pour TP2
    K_goal = 1.0
    d_safe=60
    K_obs=400

    speed = 0.1
    speed_rot = 0

    max_speed = 0.2    # Vitesse maximale en avant
    max_rot = 0.4      # Vitesse de rotation maximale

    # Constantes spéciales pour la récupération en cas de blocage
    stuck_speed = -0.2  # Reculer lorsqu'on est bloqué
    stuck_rot = 0.5     # Vitesse de rotation en mode récupération
    recovery_time = 20  # Durée (en étapes) du comportement de récupération

    # Comportement de récupération en cas de blocage
    if isStuck:
        # Compter combien de temps on reste en mode récupération
        if not hasattr(potential_field_control, 'recovery_counter'):
            potential_field_control.recovery_counter = 0
            print("Manœuvre de récupération enclenchée")
        
        if potential_field_control.recovery_counter < recovery_time:
            potential_field_control.recovery_counter += 1
            return {
                "forward": stuck_speed,
                "rotation": stuck_rot * (-1 if potential_field_control.recovery_counter % 2 else 1)
            }
        else:
            # Réinitialisation après la période de récupération
            potential_field_control.recovery_counter = 0
            isStuck = False

    dist_to_goal, grad_attr = calculate_grad_attr(current_pose, goal_pose, K_goal)

    if dist_to_goal < 6.0:
        speed = 0
        speed_rot = 0
    else:
        grad_repuls = compute_repulsive_gradient(lidar, current_pose, d_safe, K_obs)
        grad_total = grad_attr + grad_repuls

        angle_to_goal = np.arctan2(grad_total[1], grad_total[0])
        angle_diff = angle_to_goal - current_pose[2]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        speed = max_speed * np.clip(dist_to_goal, 0.0, 1.0)  # Ralentir quand on est proche
        speed_rot = np.clip(2.0 * angle_diff, -max_rot, max_rot)

        # Si on est mal orienté, d’abord tourner avant d’avancer
        if abs(angle_diff) > np.pi/4:
            speed = 0.0

    command = {"forward": speed,
               "rotation": speed_rot}

    return command
