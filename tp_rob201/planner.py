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
        """
        Return the 8 neighboring cells of the current cell that are free
        current_cell: tuple (x, y) of map coordinates
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
                self.grid.occupancy_map[nx, ny] < 0.5):  # Free cell threshold
                neighbors.append((nx, ny))

        return neighbors

    def heuristic(self, cell1, cell2):
        """
        Compute Euclidean distance between two cells
        cell1, cell2: tuples (x, y) of map coordinates
        """
        x1, y1 = cell1
        x2, y2 = cell2
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstruct path from came_from dictionary
        Returns list of poses in world coordinates
        """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        
        # Convert map coordinates to world coordinates
        world_path = []
        for cell in path:
            x_world, y_world = self.grid.conv_map_to_world(np.array([cell[0]]), np.array([cell[1]]))
            world_path.append(np.array([x_world[0], y_world[0], 0]))
        
        return world_path

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        # TODO for TP5

        # Convert world to map coordinates
        start_cell = tuple(self.grid.conv_world_to_map(start[0], start[1]))
        goal_cell = tuple(self.grid.conv_world_to_map(goal[0], goal[1]))

        # Initialize data structures
        open_set = []
        heapq.heappush(open_set, (0, start_cell))
        
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self.heuristic(start_cell, goal_cell)}

        while open_set:
            current_f_score, current = heapq.heappop(open_set)

            if current == goal_cell:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                # Distance to neighbor (using Euclidean distance)
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_cell)
                    
                    # Update or add to open_set
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # Return empty path if no path found

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
