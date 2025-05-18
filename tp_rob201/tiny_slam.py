""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        self.last_pose = None

    def _score(self, lidar, pose):
        """
        Compute the sum of occupancy values of the grid cells hit by LIDAR rays,
        for a given robot pose in world coordinates.
        """
        # 1. Obtenção dos dados do LIDAR
        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        # 2. Filtra os raios que atingiram obstáculos (distância menor que o máximo)
        mask = distances < lidar.max_range - 1e-3
        distances = distances[mask]
        angles = angles[mask]

        if len(distances) == 0:
            return 0

        # 3. Converte os pontos LIDAR para coordenadas absolutas (x, y no mundo)
        x_abs = pose[0] + distances * np.cos(pose[2] + angles)
        y_abs = pose[1] + distances * np.sin(pose[2] + angles)

        # 4. Converte para índices da grade de ocupação
        x_idx, y_idx = self.grid.conv_world_to_map(x_abs, y_abs)

        # 5. Remove os pontos fora dos limites do mapa
        valid = (x_idx >= 0) & (x_idx < self.grid.x_max_map) & \
                (y_idx >= 0) & (y_idx < self.grid.y_max_map)

        x_idx = x_idx[valid]
        y_idx = y_idx[valid]

        if len(x_idx) == 0:
            return 0

        # 6. Soma os valores das células correspondentes
        values = self.grid.occupancy_map[x_idx, y_idx]
        clamped_values = np.clip(values, 0, None)  # ignora células com valor < 0
        return np.sum(clamped_values * 0.9)  # Fator de suavização

    
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
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
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        current_pose = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar, current_pose)
        best_ref = self.odom_pose_ref.copy()
        
        # Busca local com restrições
        for _ in range(15):
            delta_angle = np.clip(np.random.normal(0, 0.01), -0.03, 0.03)
            delta_xy = np.clip(np.random.normal(0, 15, 2), -30, 30)
            
            test_ref = best_ref + np.array([delta_xy[0], delta_xy[1], delta_angle])
            test_pose = self.get_corrected_pose(raw_odom_pose, test_ref)
            current_score = self._score(lidar, test_pose)
            
            if current_score > best_score * 1.05:
                best_score = current_score
                best_ref = test_ref
        
        # Suavização conservadora
        self.odom_pose_ref = 0.3 * best_ref + 0.7 * self.odom_pose_ref
        return best_score
    
    def detect_room_change(self, lidar):
        readings = lidar.get_sensor_values()
        # Calcula a variação das leituras
        diff = np.abs(np.diff(readings))
        large_changes = np.sum(diff > 500)  # Limiar para mudança brusca (ajuste conforme necessário)
        return large_changes > len(readings)*0.3  # Se 30% das leituras tiverem grande variação

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        if self.detect_room_change(lidar):
            print("Detectada entrada em novo cômodo - ajustando mapeamento")
            # Reduz a taxa de atualização temporariamente
            self.grid.occupancy_map *= 0.9  # Suaviza o mapa existente

        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        
        # Filtro conservador
        valid = (distances < lidar.max_range * 0.9) & (distances > 100)
        distances = distances[valid]
        angles = angles[valid]
        
        if len(distances) < 15:
            return
        
        # Atualização de obstáculos (processamento em lotes pequenos)
        obs_x = pose[0] + distances * np.cos(pose[2] + angles)
        obs_y = pose[1] + distances * np.sin(pose[2] + angles)
        
        for i in range(0, len(obs_x), 5):  # Processa de 5 em 5 pontos
            batch_x = obs_x[i:i+5]
            batch_y = obs_y[i:i+5]
            self.grid.add_map_points(batch_x, batch_y, val=3)  # Valor reduzido
        
        # Atualização de espaço livre (ponto a ponto)
        for dist, angle in zip(distances, angles):
            free_dist = max(dist - 35, 30)
            # Antes de atualizar o mapa, verifique a consistência
            if abs(dist - free_dist) < 10:  # Se a distância for muito curta
                free_dist = max(dist - 40, 30)  # Usa valor mais conservador
                
            fx = pose[0] + free_dist * np.cos(pose[2] + angle)
            fy = pose[1] + free_dist * np.sin(pose[2] + angle)
            
            # Verificação explícita dos limites
            x_idx, y_idx = self.grid.conv_world_to_map(np.array([fx]), np.array([fy]))
            if (x_idx[0] >= 0 and x_idx[0] < self.grid.x_max_map and 
                y_idx[0] >= 0 and y_idx[0] < self.grid.y_max_map):
                self.grid.add_value_along_line(pose[0], pose[1], fx, fy, -0.7)
        
        # Limpeza conservadora
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -10, 10)

    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
