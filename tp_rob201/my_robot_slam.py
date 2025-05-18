"""
Robot controller definition
Complete controller including SLAM, planning, path following
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


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step display_counter to deal with init and display
        self.display_counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # Paramètres optimisés
        self.final_goal = np.array([-700,-200,0])  # Goal final
        self.waypoints = []
        self.current_goal_index = 0
        self.goal_reached_threshold = 50  # mm

        self.fixed_waypoints = np.array([
            [50, -200, 0],
            [50, -500, 0],
            [-200, -500, 0],
            [-500, -520, 0],
            [-400, -520, 0],
            [-300, -420, 0],
            [-250, -350, 0],
            [-200, -250, 0],
            [-400, -100, 0],
            [-500, -100, 0],
            [-200, 50, 0],
            [-400, -50, 0],
            [-500, -50, 0],
            [-700, -20, 0],
            [-800, -20, 0],
            [-800, -100, 0],
            [-700, -150, 0],
            [-800, -150, 0],
            [-800, -350, 0],
            [-750, -250, 0],
            [-850, -50, 0]
        ])
        
        
        # Parâmetros de localização melhorados
        self.score_history = []  # Histórico de scores
        self.max_history = 50    # Histórico maior para melhor estabilidade
        self.score_update_interval = 5  # Atualiza threshold mais frequentemente
        self.score_counter = 0
        
        # Parâmetros de movimento para atualização do mapa
        self.last_pose_for_map_update = np.array([0, 0, 0])
        self.map_update_counter = 0
        self.map_update_interval = 3  # Força atualização mais frequentemente
        
        self.corrected_pose = np.array([0, 0, 0])
        self.last_good_score = 0

        self.mapping_counter = 0
        self.position_threshold = 20  # mm

        self.last_pose_for_stuck = np.array([0, 0, 0])
        self.stuck_counter = 0
        self.max_stuck_iterations = 100
        self.last_position = None
        self.min_movement_threshold = 5  # mm


    def is_robot_stuck(self, pose):
        if pose is None:
            return False

        # Salva posição atual
        current_position = np.array(pose[:2])

        if self.occupancy_grid.obstacle_behind_wall(pose):
            self.stuck_counter += 10
            print("obstaculo atras da PAREDE")

        if self.last_position is None:
            self.last_position = current_position
            return False

        # Movimento desde a última verificação
        movement = np.linalg.norm(current_position - self.last_position)

        if movement < self.min_movement_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0  # movimento detectado, zera contador

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
        # Verifica se já atingimos todos os waypoints
        if self.current_goal_index >= len(self.fixed_waypoints):
            print("Todos os waypoints fixos foram visitados! Indo para o objetivo final.")
            return self.final_goal
        
        current_goal = self.fixed_waypoints[self.current_goal_index]
        dist_to_goal = np.linalg.norm(pose[:2] - current_goal[:2])
        
        if dist_to_goal < self.goal_reached_threshold:
            self.current_goal_index += 1
            print(f"Waypoint {self.current_goal_index} de {len(self.fixed_waypoints)} alcançado!")
            
            # Verifica se ainda há waypoints disponíveis
            if self.current_goal_index < len(self.fixed_waypoints):
                next_goal = self.fixed_waypoints[self.current_goal_index]
                print(f"Indo para o próximo waypoint em ({next_goal[0]:.1f}, {next_goal[1]:.1f})")
                return next_goal
            else:
                print("Todos os waypoints foram visitados, indo para o objetivo final")
                return self.final_goal
        
        return current_goal
    
    def get_adjusted_waypoint(self, current_goal, pose):
        """Gera um waypoint ajustado quando o robô está preso"""
        if not self.is_robot_stuck(pose):
            return current_goal  # Retorna o waypoint original se não estiver preso
        
        print("Ajustando waypoint para desatascar o robô...")
        
        # Vetor direção original
        direction = current_goal[:2] - pose[:2]
        dist_to_goal = np.linalg.norm(direction)
        
        if dist_to_goal > 0:
            direction = direction / dist_to_goal
            
            # Gera um desvio perpendicular (90° à esquerda ou direita)
            perpendicular = np.array([-direction[1], direction[0]])
            if np.random.rand() > 0.5:
                perpendicular = -perpendicular  # Aleatoriza o lado
                
            # Magnitude do desvio (20-30% da distância original)
            deviation_dist = dist_to_goal * 0.25
            new_position = pose[:2] + perpendicular * deviation_dist
            
            # Limita às fronteiras do mapa
            x_min = self.occupancy_grid.x_min_world + 100
            x_max = self.occupancy_grid.x_max_world - 100
            y_min = self.occupancy_grid.y_min_world + 100
            y_max = self.occupancy_grid.y_max_world - 100
            
            new_position[0] = np.clip(new_position[0], x_min, x_max)
            new_position[1] = np.clip(new_position[1], y_min, y_max)
            
            # Cria novo waypoint temporário
            adjusted_goal = np.array([new_position[0], new_position[1], current_goal[2]])
            print(f"Waypoint ajustado de {current_goal[:2]} para {adjusted_goal[:2]}")
            
            # Reseta o contador de stuck após ajuste
            self.stuck_counter = 0
            
            return adjusted_goal
        
        return current_goal


    def should_update_map(self, pose, current_score):
        # distance_moved = np.linalg.norm(pose[:2] - self.last_pose_for_map_update[:2])
        delta_theta = abs(pose[2] - self.last_pose_for_map_update[2])
        if delta_theta > np.pi:
            delta_theta = 2 * np.pi - delta_theta

        # Normalização do score
        if self.score_history:
            mean_score = np.mean(self.score_history[-10:])
            score_normalizado = current_score / mean_score
        else:
            score_normalizado = 1  # Permissivo no início

        threshold_fixo = 0.8  # Ajuste conforme necessário

        good_score = score_normalizado > threshold_fixo
        # moved_enough = distance_moved > 5
        not_rotating = delta_theta < 0.2
        initial_update = self.map_update_counter < 20

        if not good_score:
            print(f"Localisation incertaine (score norm: {score_normalizado:.2f} < {threshold_fixo:.2f})")

        return (good_score and not_rotating) or initial_update

    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_tp2()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar(), self.display_counter)

        if command["rotation"] > 0 and self.display_counter == 0:
            self.display_counter = random.randint(0, 10)
        elif command["rotation"] > 0 and self.display_counter != 0:
            self.display_counter -= 1


        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        mapped_percentage = self.occupancy_grid.get_mapped_area_percentage()
        
        # Condição de parada
        if mapped_percentage >= 80.0 or self.current_goal_index >= len(self.fixed_waypoints):
            return {"forward": 0.0, "rotation": 0.0}

        # Obter waypoint atual garantindo que está dentro dos limites
        current_goal = self.fixed_waypoints[min(self.current_goal_index, len(self.fixed_waypoints)-1)]

        is_stuck = self.is_robot_stuck(pose)
        
        # Atualizar goal se necessário
        dist_to_goal = np.linalg.norm(pose[:2] - current_goal[:2])
        if dist_to_goal < self.goal_reached_threshold and self.current_goal_index < len(self.fixed_waypoints)-1:
            self.current_goal_index += 1
            current_goal = self.fixed_waypoints[self.current_goal_index]
            print(f"Alcançado waypoint {self.current_goal_index}, indo para próximo")

        # Localização
        current_score = self.tiny_slam.localise(self.lidar(), pose)
        self.score_history.append(current_score)
        if len(self.score_history) > self.max_history:
            self.score_history.pop(0)

        # Atualização do mapa (somente se condições forem favoráveis)
        if self.should_update_map(pose, current_score):
            self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
            self.last_pose_for_map_update = pose.copy()
        
        # Detecção de stuck
        if is_stuck:
            print("Robô preso! Executando manobra de recuperação...")
            command = potential_field_control(self.lidar(), pose, current_goal, isStuck=True)
        else:
            command = potential_field_control(self.lidar(), pose, current_goal)

        # Exibir mapa e waypoint atual
        self.occupancy_grid.display_cv(pose, current_goal)

        # Feedback periódico
        if self.display_counter <= 0:
            print(f"Mapeado: {mapped_percentage:.1f}% | Waypoint {self.current_goal_index+1}/{len(self.fixed_waypoints)}")
            self.display_counter = 30
        else:
            self.display_counter -= 1

        return command