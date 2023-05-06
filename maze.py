import pygame
from constants import *

class Maze:
    def __init__(self, layout):
        self.layout = layout

    def get_tile_type(self, position):
        x, y = position
        return self.layout[x][y]