"""
Définition du contrôleur du robot
Contrôleur complet incluant SLAM, planification et suivi de trajectoire
"""
import numpy as np
import random

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Définition de notre contrôleur de robot
class MyRobotSlam(RobotAbstract):
    """Un contrôleur de robot incluant SLAM, planification de trajectoire et suivi de trajectoire"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passage des paramètres à la classe parente
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # Compteur d'affichage pour gérer l'init et l'affichage
        self.display_counter = 0

        # Initialisation de l'objet SLAM
        # Ici, on triche pour obtenir une taille de grille d'occupation pas trop grande,
        # en utilisant la position de départ du robot et la taille maximale de la carte
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)
        
        self.fixed_waypoints = np.array([
            [50, -200, 0],
            [50, -500, 0],
            [-200, -500, 0],
            [-500, -520, 0],
            [-350, -520, 0],
            [-300, -420, 0],
            [-250, -350, 0],
            [-200, -250, 0],
            [-400, -100, 0],
            [-500, -100, 0],
            [-500, -50, 0],
            [-200, 50, 0],
            [-400, -50, 0],
            [-500, -50, 0],
            [-700, -20, 0],
            [-800, -20, 0],
            [-800, -100, 0],
            [-650, -100, 0],
            [-800, -100, 0],
            [-850, -350, 0],
            [-800, -250, 0],
            [-900, -300, 0],
            [-900, -50, 0]
        ])
        
        if self.occupancy_grid.load("map"):  # Charge une carte enregistrée
            self.current_goal_index = len(self.fixed_waypoints)
            self.need_to_save_map = False
        else:
            self.current_goal_index = 0
            self.need_to_save_map = True

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # Paramètres optimisés
        self.final_goal = np.array([-800,-300,0])  # Objectif final
        self.waypoints = []
        
        self.goal_reached_threshold = 50  # mm
        
        # Paramètres de localisation améliorés
        self.score_history = []  # Historique des scores
        self.max_history = 50    # Historique plus long pour meilleure stabilité
        self.score_update_interval = 5  # Mise à jour plus fréquente du seuil
        self.score_counter = 0
        
        # Paramètres de mouvement pour mise à jour de la carte
        self.last_pose_for_map_update = np.array([0, 0, 0])
        self.map_update_counter = 0
        self.map_update_interval = 3  # Force une mise à jour plus fréquente
        
        self.corrected_pose = np.array([0, 0, 0])
        self.last_good_score = 0

        self.mapping_counter = 0
        self.position_threshold = 20  # mm

        self.last_pose_for_stuck = np.array([0, 0, 0])
        self.stuck_counter = 0
        self.max_stuck_iterations = 100
        self.last_position = None
        self.min_movement_threshold = 5  # mm

        self.iteration_count = 0
        self.path = []
        self.path_following = False
        self.return_goal = np.array([0, 0, 0])
        self.exploration_mode = True
        self.reached_final_goal = False

    def is_robot_stuck(self, pose):
        if pose is None:
            return False

        # Sauvegarde de la position actuelle
        current_position = np.array(pose[:2])

        if self.occupancy_grid.obstacle_behind_wall(pose):
            self.stuck_counter += 10
            print("obstacle derrière un MUR")

        if self.last_position is None:
            self.last_position = current_position
            return False

        # Mouvement depuis la dernière vérification
        movement = np.linalg.norm(current_position - self.last_position)

        if movement < self.min_movement_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0  # mouvement détecté, réinitialise le compteur

        self.last_position = current_position

        if self.stuck_counter >= self.max_stuck_iterations:
            self.stuck_counter = 0
            return True

        return False
    
    # def generate_randomized_zigzag_waypoints(self, grid_resolution=200, noise=50):
    #     """
    #     Gera waypoints em zigue-zague cobrindo o mapa, com leve aleatoriedade para variar entre execuções.
        
    #     Args:
    #         grid_resolution (int): distância entre linhas/colunas da grade (em mm).
    #         noise (int): ruído máximo (em mm) adicionado para aleatoriedade nos waypoints.
            
    #     Returns:
    #         List[np.ndarray]: lista de waypoints cobrindo o mapa.
    #     """
    #     margin = 150  # Evita encostar nas bordas
    #     x_min = self.occupancy_grid.x_min_world + margin
    #     x_max = self.occupancy_grid.x_max_world - margin
    #     y_min = self.occupancy_grid.y_min_world + margin
    #     y_max = self.occupancy_grid.y_max_world - margin

    #     xs = np.arange(x_min, x_max, grid_resolution)
    #     ys = np.arange(y_min, y_max, grid_resolution)

    #     waypoints = []

    #     for i, y in enumerate(ys):
    #         # Alterna direção para zigue-zague
    #         x_sequence = xs if i % 2 == 0 else xs[::-1]

    #         for x in x_sequence:
    #             # Adiciona ruído para variar entre execuções
    #             dx = random.uniform(-noise, noise)
    #             dy = random.uniform(-noise, noise)
    #             x_r = np.clip(x + dx, x_min, x_max)
    #             y_r = np.clip(y + dy, y_min, y_max)
    #             wp = np.array([x_r, y_r, 0])
    #             waypoints.append(wp)

    #     print(f"Gerados {len(waypoints)} waypoints em zigue-zague com aleatoriedade.")
    #     return waypoints


    # def generate_systematic_waypoints(self, max_attempts=10):
    #     """Generate a systematic waypoint for exploration, dividing the map into quadrants"""
    #     pose = self.odometer_values()
        
    #     # Define os limites do mapa com margem
    #     margin = 200
    #     x_min = self.occupancy_grid.x_min_world + margin
    #     x_max = self.occupancy_grid.x_max_world - margin
    #     y_min = self.occupancy_grid.y_min_world + margin
    #     y_max = self.occupancy_grid.y_max_world - margin
        
    #     # Divide o mapa em quadrantes
    #     x_mid = (x_min + x_max) / 2
    #     y_mid = (y_min + y_max) / 2
        
    #     # Define os quadrantes
    #     quadrants = [
    #         (x_min, x_mid, y_min, y_mid),      # Quadrante inferior esquerdo
    #         (x_mid, x_max, y_min, y_mid),      # Quadrante inferior direito
    #         (x_min, x_mid, y_mid, y_max),      # Quadrante superior esquerdo
    #         (x_mid, x_max, y_mid, y_max)       # Quadrante superior direito
    #     ]
        
    #     # Conta quantos waypoints já foram visitados em cada quadrante
    #     quadrant_counts = [0, 0, 0, 0]
    #     for wp in self.waypoints:
    #         x, y = wp[0], wp[1]
    #         if x < x_mid and y < y_mid:
    #             quadrant_counts[0] += 1
    #         elif x >= x_mid and y < y_mid:
    #             quadrant_counts[1] += 1
    #         elif x < x_mid and y >= y_mid:
    #             quadrant_counts[2] += 1
    #         else:
    #             quadrant_counts[3] += 1
        
    #     # Escolhe o quadrante com menos pontos visitados
    #     target_quadrant = np.argmin(quadrant_counts)
    #     x_min_q, x_max_q, y_min_q, y_max_q = quadrants[target_quadrant]
        
    #     # Gera pontos dentro do quadrante escolhido
    #     for attempt in range(max_attempts):
    #         # Gera um ponto aleatório dentro do quadrante
    #         target_x = np.random.uniform(x_min_q, x_max_q)
    #         target_y = np.random.uniform(y_min_q, y_max_q)
            
    #         waypoint = np.array([target_x, target_y, 0])
            
    #         # Verifica se o waypoint já foi visitado
    #         min_distance_to_visited = float('inf')
    #         for visited in self.waypoints:
    #             dist = np.linalg.norm(waypoint[:2] - visited[:2])
    #             min_distance_to_visited = min(min_distance_to_visited, dist)
            
    #         # Se o waypoint estiver muito próximo de um já visitado, tenta novamente
    #         if min_distance_to_visited < 100:  # Aumentado para 100mm para melhor cobertura
    #             print(f"Waypoint muito próximo de um já visitado, tentativa {attempt + 1}/{max_attempts}")
    #             continue
            
    #         print(f"Generated new waypoint in quadrant {target_quadrant + 1} at ({target_x:.1f}, {target_y:.1f})")
    #         return waypoint
        
    #     # Se não conseguir gerar um waypoint válido, tenta em outro quadrante
    #     print("Não foi possível gerar um waypoint válido no quadrante escolhido. Tentando outro quadrante...")
    #     # Ordena os quadrantes por número de visitas
    #     sorted_quadrants = np.argsort(quadrant_counts)
        
    #     # Tenta o segundo quadrante menos visitado
    #     if len(sorted_quadrants) > 1:
    #         second_least_visited = sorted_quadrants[1]
    #         x_min_q, x_max_q, y_min_q, y_max_q = quadrants[second_least_visited]
            
    #         target_x = np.random.uniform(x_min_q, x_max_q)
    #         target_y = np.random.uniform(y_min_q, y_max_q)
            
    #         waypoint = np.array([target_x, target_y, 0])
    #         print(f"Generated fallback waypoint in quadrant {second_least_visited + 1} at ({target_x:.1f}, {target_y:.1f})")
    #         return waypoint
        
    #     # Se tudo falhar, gera um ponto aleatório em qualquer lugar do mapa
    #     target_x = np.random.uniform(x_min, x_max)
    #     target_y = np.random.uniform(y_min, y_max)
    #     waypoint = np.array([target_x, target_y, 0])
    #     print(f"Generated emergency waypoint at ({target_x:.1f}, {target_y:.1f})")
    #     return waypoint

    # def generate_grid_waypoints(self, grid_size=6):
    #     """Gera uma grade regular de waypoints dentro dos limites do mapa"""
    #     margin = 200
    #     x_min = self.occupancy_grid.x_min_world + margin
    #     x_max = self.occupancy_grid.x_max_world - margin
    #     y_min = self.occupancy_grid.y_min_world + margin
    #     y_max = self.occupancy_grid.y_max_world - margin

    #     xs = np.linspace(x_min, x_max, grid_size)
    #     ys = np.linspace(y_min, y_max, grid_size)
    #     grid_points = []
    #     for x in xs:
    #         for y in ys:
    #             grid_points.append(np.array([x, y, 0]))
    #     return grid_points

    # def get_next_grid_waypoint(self):
    #     """Retorna o waypoint da grade mais próximo ainda não visitado"""
    #     if not hasattr(self, 'grid_waypoints'):
    #         self.grid_waypoints = self.generate_grid_waypoints(grid_size=6)
    #         self.visited_waypoints = set()
    #     pose = self.odometer_values()
    #     # Filtra waypoints já visitados
    #     unvisited = [tuple(wp) for wp in self.grid_waypoints if tuple(wp) not in self.visited_waypoints]
    #     if not unvisited:
    #         print("Todos os waypoints da grade foram visitados!")
    #         return self.final_goal
    #     # Escolhe o mais próximo
    #     dists = [np.linalg.norm(np.array(wp)[:2] - pose[:2]) for wp in unvisited]
    #     idx = np.argmin(dists)
    #     next_wp = np.array(unvisited[idx])
    #     return next_wp

    def mark_waypoint_visited(self, waypoint):
        if not hasattr(self, 'visited_waypoints'):
            self.visited_waypoints = set()
        self.visited_waypoints.add(tuple(waypoint))

    def exploration_mode(self, pose):
        # Vérifie si tous les waypoints ont été atteints
        if self.current_goal_index >= len(self.fixed_waypoints):
            print("Tous les waypoints fixes ont été visités ! Direction objectif final.")
            return self.final_goal
        
        current_goal = self.fixed_waypoints[self.current_goal_index]
        dist_to_goal = np.linalg.norm(pose[:2] - current_goal[:2])
        
        if dist_to_goal < self.goal_reached_threshold:
            self.current_goal_index += 1
            print(f"Waypoint {self.current_goal_index} sur {len(self.fixed_waypoints)} atteint !")
            
            # Vérifie s'il reste des waypoints
            if self.current_goal_index < len(self.fixed_waypoints):
                next_goal = self.fixed_waypoints[self.current_goal_index]
                print(f"Aller au prochain waypoint à ({next_goal[0]:.1f}, {next_goal[1]:.1f})")
                return next_goal
            else:
                print("Tous les waypoints ont été visités, direction objectif final")
                return self.final_goal
        
        return current_goal

    def should_update_map(self, pose, current_score):
        # distance_moved = np.linalg.norm(pose[:2] - self.last_pose_for_map_update[:2])
        delta_theta = abs(pose[2] - self.last_pose_for_map_update[2])
        if delta_theta > np.pi:
            delta_theta = 2 * np.pi - delta_theta

        # Normalisation du score
        if self.score_history:
            mean_score = np.mean(self.score_history[-10:])
            score_normalisé = current_score / mean_score
        else:
            score_normalisé = 1  # Tolérant au début

        seuil_fixe = 0.85  # Ajuster si besoin

        good_score = score_normalisé > seuil_fixe
        # moved_enough = distance_moved > 5
        not_rotating = delta_theta < 0.2
        initial_update = self.map_update_counter < 20

        if not good_score:
            print(f"Localisation incertaine (score norm: {score_normalisé:.2f} < {seuil_fixe:.2f})")

        return (good_score and not_rotating) or initial_update
    
    def follow_path(self, pose):
        if not self.path:
            print("Aucun chemin à suivre !")
            return {"forward": 0, "rotation": 0}
        current_goal = self.path[0]
        dist_to_goal = np.linalg.norm(pose[:2] - current_goal[:2])
        if dist_to_goal < self.goal_reached_threshold:
            self.path.pop(0)
            if not self.path:
                print("Destination du chemin atteinte !")
                return {"forward": 0, "rotation": 0}
            current_goal = self.path[0]
            print(f"Aller au prochain point du chemin à ({current_goal[0]:.1f}, {current_goal[1]:.1f})")
        is_stuck = self.is_robot_stuck(pose)
        command = potential_field_control(self.lidar(), pose, current_goal, isStuck=is_stuck)
        return command
        

    def control(self):
        """
        Fonction principale de contrôle exécutée à chaque pas de temps
        """
        return self.control_tp2()

    def control_tp1(self):
        """
        Fonction de contrôle pour le TP1
        Fonction de contrôle avec un mouvement aléatoire minimal
        """
        self.tiny_slam.compute()

        # Calculer une nouvelle commande de vitesse pour éviter les obstacles
        command = reactive_obst_avoid(self.lidar(), self.display_counter)

        if command["rotation"] > 0 and self.display_counter == 0:
            self.display_counter = random.randint(0, 10)
        elif command["rotation"] > 0 and self.display_counter != 0:
            self.display_counter -= 1

        return command

    def control_tp2(self):
        """
        Fonction de contrôle pour le TP2
        Fonction principale avec SLAM complet, exploration aléatoire et planification de chemin
        """
        pose = self.odometer_values()
        mapped_percentage = self.occupancy_grid.get_mapped_area_percentage()
        
        # Condition d'arrêt
        if mapped_percentage >= 46 and self.current_goal_index+1 >= len(self.fixed_waypoints):
            if self.need_to_save_map:
                self.occupancy_grid.save("map")
                self.need_to_save_map = False
            current_goal = self.final_goal
            self.exploration_mode = False

        if self.exploration_mode:

            # Obtenir le waypoint actuel en s'assurant qu'il est dans les limites
            current_goal = self.fixed_waypoints[min(self.current_goal_index, len(self.fixed_waypoints)-1)]

            is_stuck = self.is_robot_stuck(pose)
            
            # Mettre à jour le goal si nécessaire
            dist_to_goal = np.linalg.norm(pose[:2] - current_goal[:2])
            if dist_to_goal < self.goal_reached_threshold and self.current_goal_index < len(self.fixed_waypoints)-1:
                self.current_goal_index += 1
                current_goal = self.fixed_waypoints[self.current_goal_index]
                print(f"Waypoint {self.current_goal_index} atteint, passage au suivant")

            # Localisation
            current_score = self.tiny_slam.localise(self.lidar(), pose)
            self.score_history.append(current_score)
            if len(self.score_history) > self.max_history:
                self.score_history.pop(0)

            # Mise à jour de la carte (seulement si les conditions sont favorables)
            if self.should_update_map(pose, current_score):
                self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)
                self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
                self.last_pose_for_map_update = pose.copy()
            
            # Détection de blocage
            if is_stuck:
                print("Robot bloqué ! Exécution d'une manœuvre de récupération...")
                command = potential_field_control(self.lidar(), pose, current_goal, isStuck=True)
            else:
                command = potential_field_control(self.lidar(), pose, current_goal)
        
        # Phase 3 : Planifier et suivre le chemin vers le goal final
        if not self.exploration_mode and not self.reached_final_goal and not self.path_following:
            print("Exploration terminée ! Planification du chemin vers le goal final (-700, -200, 0)")
            self.path = self.planner.plan(pose, self.final_goal)
            if not self.path:
                print("Aucun chemin trouvé vers le goal final ! Reprise de l'exploration...")
                self.exploration_mode = True
                self.iteration_count -= 1
                current_goal = self.exploration_mode(pose)
                command = potential_field_control(self.lidar(), pose, current_goal)
                self.occupancy_grid.display_cv(pose, current_goal)
                return command
            self.path_following = True
            print(f"Chemin planifié vers le goal final avec {len(self.path)} waypoints")
            traj_array = np.array([pose[:2] for pose in self.path]).T
            self.occupancy_grid.display_cv(pose, self.final_goal, traj=traj_array)
            return self.follow_path(pose)

        # Phase 4 : Vérifier si le goal final est atteint et planifier le retour à l'origine
        if self.path_following and not self.reached_final_goal:
            dist_to_final = np.linalg.norm(pose[:2] - self.final_goal[:2])
            if dist_to_final < self.goal_reached_threshold or not self.path:
                print("Goal final atteint ! Planification du retour à l'origine (0, 0, 0)")
                self.reached_final_goal = True
                self.path_following = False
                self.path = self.planner.plan(pose, self.return_goal)
                if not self.path:
                    print("Aucun chemin trouvé vers l'origine ! Arrêt du robot.")
                    self.occupancy_grid.save("final_map")
                    return {"forward": 0, "rotation": 0}
                print(f"Chemin planifié vers l'origine avec {len(self.path)} waypoints")
                traj_array = np.array([pose[:2] for pose in self.path]).T
                if traj_array.shape[0] == 2:  # Vérifier que c'est bien un tableau 2D
                    self.occupancy_grid.display_cv(pose, self.return_goal, traj=traj_array)
                else:
                    self.occupancy_grid.display_cv(pose, self.return_goal)
                return self.follow_path(pose)
            command = self.follow_path(pose)
            if self.path and len(self.path) > 0:
                traj_array = np.array([pose[:2] for pose in self.path]).T
                if traj_array.shape[0] == 2:
                    self.occupancy_grid.display_cv(pose, self.final_goal, traj=traj_array)
                else:
                    self.occupancy_grid.display_cv(pose, self.final_goal)
            else:
                self.occupancy_grid.display_cv(pose, self.final_goal)
            return command

        # Phase 5 : Suivre le chemin vers l'origine
        if self.reached_final_goal:
            dist_to_origin = np.linalg.norm(pose[:2] - self.return_goal[:2])
            if dist_to_origin < self.goal_reached_threshold:
                print("Origine atteinte ! Arrêt du robot.")
                self.occupancy_grid.save("final_map")
                return {"forward": 0, "rotation": 0}
            if not self.path_following:
                self.path_following = True
            command = self.follow_path(pose)
            if self.path and len(self.path) > 0:
                traj_array = np.array([pose[:2] for pose in self.path]).T
                if traj_array.shape[0] == 2:
                    self.occupancy_grid.display_cv(pose, self.return_goal, traj=traj_array)
                else:
                    self.occupancy_grid.display_cv(pose, self.return_goal)
            else:
                self.occupancy_grid.display_cv(pose, self.return_goal)
            return command

        # Afficher la carte et le waypoint actuel
        self.occupancy_grid.display_cv(pose, current_goal)

        # Feedback périodique
        if self.display_counter <= 0:
            print(f"Cartographié : {mapped_percentage:.1f}% | Waypoint {self.current_goal_index+1}/{len(self.fixed_waypoints)}")
            self.display_counter = 30
        else:
            self.display_counter -= 1

        return command