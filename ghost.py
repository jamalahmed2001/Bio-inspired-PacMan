import pygame
from constants import *

class Ghost:
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color
        self.radius = GHOST_RADIUS
        self.speed = GHOST_SPEED

    def move(self, dx, dy):
        self.pos = (self.pos[0] + dx * self.speed, self.pos[1] + dy * self.speed)

    def update(self):
        pass

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), self.radius)

    def get_position(self):
        return self.pos