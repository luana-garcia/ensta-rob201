"""
Classe Planner
Implémentation de l'algorithme A*
"""

import numpy as np
import heapq
from scipy.ndimage import binary_dilation

from occupancy_grid import OccupancyGrid

class Planner:
    """Planificateur simple basé sur une grille d'occupation"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.FREE_CELL = 0.8  # Augmenté pour éviter les cellules proches des murs

        # Création d'une carte d'occupation dilatée
        self.dilated_map = np.copy(self.grid.occupancy_map)
        walls = self.dilated_map > 45  # Seuil pour détecter les murs
        walls_dilation = binary_dilation(walls, iterations=12)
        self.dilated_map[walls_dilation] = 50  # Les zones dilatées ont une forte occupation

        # Origine du repère odom dans le repère de la carte
        self.odom_pose_ref = np.array([0, 0, 0])

    def get_neighbors(self, current_cell):
        """
        Retourne les 8 cellules voisines de la cellule actuelle qui sont libres
        current_cell : tuple (x, y) en coordonnées de la carte
        """
        x, y = current_cell
        neighbors = []

        directions = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),         (0, 1),
                     (1, -1), (1, 0), (1, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.grid.x_max_map and 
                0 <= ny < self.grid.y_max_map and
                self.grid.occupancy_map[nx, ny] < 0.5):  # Seuil pour cellule libre
                neighbors.append((nx, ny))

        return neighbors

    def heuristic(self, cell1, cell2):
        """
        Calcule la distance euclidienne entre deux cellules
        cell1, cell2 : tuples (x, y) en coordonnées de la carte
        """
        x1, y1 = cell1
        x2, y2 = cell2
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstruit le chemin à partir du dictionnaire came_from
        Retourne une liste de poses en coordonnées du monde
        """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        
        # Conversion des coordonnées carte vers monde
        world_path = []
        for cell in path:
            x_world, y_world = self.grid.conv_map_to_world(np.array([cell[0]]), np.array([cell[1]]))
            world_path.append(np.array([x_world[0], y_world[0], 0]))
        
        return world_path

    def plan(self, start, goal):
        """
        Calcule un chemin en utilisant A*, recalcule le plan si le départ ou l’arrivée change
        start : [x, y, theta] nparray, position initiale en coordonnées du monde (theta non utilisé)
        goal : [x, y, theta] nparray, position cible en coordonnées du monde (theta non utilisé)
        """
        # Convertir les coordonnées du monde en coordonnées carte
        start_cell = tuple(self.grid.conv_world_to_map(start[0], start[1]))
        goal_cell = tuple(self.grid.conv_world_to_map(goal[0], goal[1]))

        # Vérifier si le départ ou l’arrivée sont dans des obstacles
        if (self.dilated_map[start_cell[0], start_cell[1]] >= self.FREE_CELL or
            self.dilated_map[goal_cell[0], goal_cell[1]] >= self.FREE_CELL):
            print("Départ ou arrivée dans un obstacle (carte dilatée), impossible de planifier un chemin !")
            return []

        # Initialiser les structures de données
        open_set = []
        heapq.heappush(open_set, (0, start_cell))
        
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self.heuristic(start_cell, goal_cell)}

        while open_set:
            current_f_score, current = heapq.heappop(open_set)

            if current == goal_cell:
                path = self.reconstruct_path(came_from, current)
                print(f"Chemin planifié avec {len(path)} points de passage")
                return path

            for neighbor in self.get_neighbors(current):
                # Distance vers le voisin (distance euclidienne)
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_cell)
                    
                    # Mettre à jour ou ajouter au open_set
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("Aucun chemin trouvé !")
        return []

    def explore_frontiers(self):
        """ Exploration basée sur les frontières """
        goal = np.array([0, 0, 0])  # Frontière à atteindre pour l'exploration
        return goal
