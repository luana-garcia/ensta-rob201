"""
Planner class
Implementation of A*
"""

import numpy as np
import heapq

from occupancy_grid import OccupancyGrid

class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid


        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def get_neighbors(self, current_cell):
        x, y = current_cell
        neighbors = []

        directions = [(-1, -1), (-1, 0), (-1, 1),
                    ( 0, -1),          ( 0, 1),
                    ( 1, -1), ( 1, 0), ( 1, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map.shape[0] and 0 <= ny < self.map.shape[1]:
                if self.map[nx, ny] < 0.5:  # limiar para considerar livre
                    neighbors.append((nx, ny))

        return neighbors

    def heuristic(self, cell1, cell2):
        x1, y1 = cell1
        x2, y2 = cell2
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        # TODO for TP5

        path = [start, goal]  # list of poses

        # Conversão: coordenadas do mundo -> células da grade
        start_cell = self.grid.conv_world_to_map(start[0], start[1])
        goal_cell = self.grid.conv_world_to_map(goal[0], goal[1])

        # Inicializações
        open_set = []
        heapq.heappush(open_set, (0, start_cell))

        came_from = {}  # caminho reconstruído
        g_score = {start_cell: 0}
        f_score = {start_cell: self.heuristic(start_cell, goal_cell)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_cell:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.heuristic(current, neighbor)

                if tentative_g < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_cell)
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # falha

        return path

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
