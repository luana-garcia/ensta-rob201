""" Un code simple de navigation robotique incluant SLAM, exploration, planification"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """SLAM simple avec grille d’occupation"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origine du repère odométrique dans le repère carte
        self.odom_pose_ref = np.array([0, 0, 0])
        self.last_pose = None

    def _score(self, lidar, pose):
        """
        Calcule la somme des valeurs d’occupation des cellules de la grille touchées par les rayons LIDAR,
        pour une pose donnée du robot dans les coordonnées du monde.
        """
        # 1. Obtention des données du LIDAR
        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        # 2. Filtre les rayons qui ont touché des obstacles (distance inférieure au maximum)
        mask = distances < lidar.max_range - 1e-3
        distances = distances[mask]
        angles = angles[mask]

        if len(distances) == 0:
            return 0

        # 3. Convertit les points LIDAR en coordonnées absolues (x, y dans le monde)
        x_abs = pose[0] + distances * np.cos(pose[2] + angles)
        y_abs = pose[1] + distances * np.sin(pose[2] + angles)

        # 4. Convertit en indices de la grille d’occupation
        x_idx, y_idx = self.grid.conv_world_to_map(x_abs, y_abs)

        # 5. Supprime les points hors des limites de la carte
        valid = (x_idx >= 0) & (x_idx < self.grid.x_max_map) & \
                (y_idx >= 0) & (y_idx < self.grid.y_max_map)

        x_idx = x_idx[valid]
        y_idx = y_idx[valid]

        if len(x_idx) == 0:
            return 0

        # 6. Somme les valeurs des cellules correspondantes
        values = self.grid.occupancy_map[x_idx, y_idx]
        clamped_values = np.clip(values, 0, None)  # ignore les cellules avec valeur < 0
        return np.sum(clamped_values * 0.9)  # Facteur de lissage

    
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Calcule la pose corrigée dans le repère carte à partir de la pose odométrique brute + pose du repère odom,
        soit donnée comme second paramètre, soit utilisée depuis l’objet
        odom : position odométrique brute
        odom_pose_ref : optionnel, origine du repère odométrique si donnée,
                        utilise self.odom_pose_ref sinon
        """
        # TODO pour TP4
        corrected_pose = odom_pose

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
        theta_ref = odom_pose_ref[2]
        R = np.array([
            [np.cos(theta_ref), -np.sin(theta_ref)],
            [np.sin(theta_ref),  np.cos(theta_ref)]
        ])
        
        corrected_xy = R @ odom_pose[:2] + odom_pose_ref[:2]
        corrected_theta = self.normalize_angle(odom_pose[2] + odom_pose_ref[2])
        
        corrected_pose = np.array([corrected_xy[0], corrected_xy[1], corrected_theta])

        return corrected_pose

    def localise(self, lidar, raw_odom_pose, N = 10, sigma=np.array([0.08, 0.08, 0.03])):
        """
        Calcule la position du robot par rapport à la carte, et met à jour la référence odométrique
        lidar : objet placebot avec les données lidar
        odom : [x, y, theta] nparray, position odométrique brute
        """
        # TODO pour TP4

        current_pose = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar, current_pose)
        best_ref = self.odom_pose_ref.copy()
        
        # Recherche locale avec contraintes
        for _ in range(15):
            delta_angle = np.clip(np.random.normal(0, 0.01), -0.03, 0.03)
            delta_xy = np.clip(np.random.normal(0, 15, 2), -30, 30)
            
            test_ref = best_ref + np.array([delta_xy[0], delta_xy[1], delta_angle])
            test_pose = self.get_corrected_pose(raw_odom_pose, test_ref)
            current_score = self._score(lidar, test_pose)
            
            if current_score > best_score * 1.05:
                best_score = current_score
                best_ref = test_ref
        
        # Lissage conservateur
        self.odom_pose_ref = 0.3 * best_ref + 0.7 * self.odom_pose_ref
        return best_score
    
    def detect_room_change(self, lidar):
        readings = lidar.get_sensor_values()
        # Calcule la variation des lectures
        diff = np.abs(np.diff(readings))
        large_changes = np.sum(diff > 500)  # Seuil pour changement brusque (ajuster si nécessaire)
        return large_changes > len(readings)*0.3  # Si 30% des lectures présentent une grande variation

    def update_map(self, lidar, pose):
        """
        Mise à jour bayésienne de la carte avec une nouvelle observation
        lidar : objet placebot avec les données lidar
        pose : [x, y, theta] nparray, pose corrigée dans les coordonnées du monde
        """
        # TODO pour TP3
        if self.detect_room_change(lidar):
            print("Entrée détectée dans une nouvelle pièce - ajustement de la cartographie")
            # Réduit temporairement le taux de mise à jour
            self.grid.occupancy_map *= 0.9  # Lisse la carte existante

        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        
        # Filtrage conservateur
        valid = (distances < lidar.max_range * 0.9) & (distances > 100)
        distances = distances[valid]
        angles = angles[valid]
        
        if len(distances) < 15:
            return
        
        # Mise à jour des obstacles (traitement par petits lots)
        obs_x = pose[0] + distances * np.cos(pose[2] + angles)
        obs_y = pose[1] + distances * np.sin(pose[2] + angles)
        
        for i in range(0, len(obs_x), 5):  # Traite les points 5 par 5
            batch_x = obs_x[i:i+5]
            batch_y = obs_y[i:i+5]
            self.grid.add_map_points(batch_x, batch_y, val=3)  # Valeur réduite
        
        # Mise à jour de l’espace libre (point par point)
        for dist, angle in zip(distances, angles):
            free_dist = max(dist - 35, 30)
            # Avant de mettre à jour la carte, vérifie la cohérence
            if abs(dist - free_dist) < 10:  # Si la distance est trop courte
                free_dist = max(dist - 40, 30)  # Utilise une valeur plus conservatrice
                
            fx = pose[0] + free_dist * np.cos(pose[2] + angle)
            fy = pose[1] + free_dist * np.sin(pose[2] + angle)
            
            # Vérification explicite des limites
            x_idx, y_idx = self.grid.conv_world_to_map(np.array([fx]), np.array([fy]))
            if (x_idx[0] >= 0 and x_idx[0] < self.grid.x_max_map and 
                y_idx[0] >= 0 and y_idx[0] < self.grid.y_max_map):
                self.grid.add_value_along_line(pose[0], pose[1], fx, fy, -0.7)
        
        # Nettoyage conservateur
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -10, 10)

    def compute(self):
        """ Fonction inutile, juste pour l’exercice sur l’utilisation du profileur """
        # Supprimer après TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Mauvaise implémentation de la conversion polaire vers cartésienne
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
