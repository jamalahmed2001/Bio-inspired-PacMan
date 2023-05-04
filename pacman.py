import pygame
import sys

# Game Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
PACMAN_SIZE = 30
VELOCITY = 5

# PacMan 
class PacMan:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = VELOCITY

    def draw(self, screen):
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), PACMAN_SIZE // 2)

    def userControl(self,pygame,room):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.move(-1, 0, room)
        elif keys[pygame.K_RIGHT]:
            self.move(1, 0, room)
        elif keys[pygame.K_UP]:
            self.move(0, -1, room)
        elif keys[pygame.K_DOWN]:
            self.move(0, 1, room)

    def move(self, dx, dy, room):
        new_x = self.x + dx * self.velocity
        new_y = self.y + dy * self.velocity
        if not self.roomCollision(new_x, new_y, room):
            self.x = new_x
            self.y = new_y

    def roomCollision(self, x, y, room):
        pacman_rect = pygame.Rect(x - PACMAN_SIZE // 2, y - PACMAN_SIZE // 2, PACMAN_SIZE, PACMAN_SIZE)
        room_rect = pygame.Rect(room.x, room.y, room.width, room.height)
        return not room_rect.contains(pacman_rect)

# Room class
class Room:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

